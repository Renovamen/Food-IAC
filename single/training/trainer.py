import time
from torch.nn.utils.rnn import pack_padded_sequence
from utils.common import *
from metrics import Metrics
from .tensorboard import TensorboardWriter

class Trainer:
    '''
    an encoder-decoder pipeline
    Tearcher Forcing is used during training and validation

    input params:
        epochs: we should train the model for __ epochs
        device: use GPU or not
        start_epoch: we should start training the model from __th epoch
        epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
        best_bleu4: best BLEU-4 score until now
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        encoder: encoder (based on CNN)
        decoder: decoder (based on LSTM)
        encoder_optimizer: optimizer for encoder (Adam) (if fine-tune)
        decoder_optimizer: optimizer for decoder (Adam)
        loss_function: loss function (cross entropy)
        grad_clip: gradient threshold in clip gradients
        fine_tune_encoder: fine-tune encoder or not
        tensorboard: enable tensorboard or not?
        log_dir (str): folder for saving logs for tensorboard
    '''

    def __init__(self, epochs, device, word_map, rev_word_map,
                    start_epoch, epochs_since_improvement, best_bleu4,
                    train_loader, val_loader,
                    encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    loss_function, grad_clip, fine_tune_encoder,
                    tensorboard = False, log_dir = None):

        self.device = device  # GPU / CPU

        self.epochs = epochs
        self.word_map = word_map
        self.rev_word_map = rev_word_map

        self.start_epoch = start_epoch
        self.epochs_since_improvement = epochs_since_improvement
        self.best_bleu4 = best_bleu4

        self.train_loader =  train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.loss_function = loss_function

        self.grad_clip = grad_clip
        self.fine_tune_encoder = fine_tune_encoder

        self.print_freq = 100  # print training/validation stats every __ batches
        # setup visualization writer instance
        self.writer = TensorboardWriter(log_dir, tensorboard)
        self.len_epoch = len(self.train_loader)

    def train(self, epoch):
        '''
        train an epoch

        input params:
            epoch: current epoch num
        '''

        self.decoder.train()  # train mode (dropout and batchnorm is used)
        self.encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter(tag = 'loss', writer = self.writer)  # loss (per word decoded)
        top5accs = AverageMeter(tag = 'top5acc', writer = self.writer)  # top5 accuracy

        start = time.time()

        # batches
        for i, (imgs, caps, caplens) in enumerate(self.train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            # forward encoder
            imgs = self.encoder(imgs)

            # forward decoder
            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(imgs, caps, caplens)

            # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)[0]

            # calc loss
            loss = self.loss_function(scores, targets)

            # clear gradient of last batch
            self.decoder_optimizer.zero_grad()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()

            # backward
            loss.backward()

            # clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.decoder_optimizer, self.grad_clip)
                if self.encoder_optimizer is not None:
                    clip_gradient(self.encoder_optimizer, self.grad_clip)

            # update weights
            self.decoder_optimizer.step()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()

            # set step for tensorboard
            step = (epoch - 1) * self.len_epoch + i
            self.writer.set_step(step = step, mode = 'train')

            # keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # print status
            if i % self.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(self.train_loader),
                                                                            batch_time = batch_time,
                                                                            data_time = data_time,
                                                                            loss = losses,
                                                                            top5 = top5accs)
                )

    def validate(self):
        '''
        validate an epoch

        input params:
            val_loader: DataLoader for validation data
            encoder: an encoder (based on CNN)
            decoder: a decoder (based on LSTM)
            loss_function: loss function (cross entropy)

        return:
            bleu4: BLEU-4 score
        '''

        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        if self.encoder is not None:
            self.encoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        start = time.time()

        ground_truth = list()  # ground_truth (true captions) for calculating BLEU-4 score
        prediction = list()  # prediction (predicted captions)

        # explicitly disable gradient calculation to avoid CUDA memory error
        # solves the issue #57
        with torch.no_grad():
            # Batches
            for i, (imgs, caps, caplens, allcaps) in enumerate(self.val_loader):

                # move to device, if available
                imgs = imgs.to(self.device)
                caps = caps.to(self.device)
                caplens = caplens.to(self.device)

                # forward encoder
                if self.encoder is not None:
                    imgs = self.encoder(imgs)

                # forward decoder
                scores, caps_sorted, decode_lengths, sort_ind = self.decoder(imgs, caps, caplens)

                # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)[0]

                # calc loss
                loss = self.loss_function(scores, targets)

                # keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % self.print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(self.val_loader),
                                                                                    batch_time = batch_time,
                                                                                    loss = losses,
                                                                                    top5 = top5accs)
                    )

                # store ground truth captions and predicted captions of each image
                # for n images, each of them has one prediction and multiple ground truths (a, b, c...):
                # prediction = [ [hyp1], [hyp2], ..., [hypn] ]
                # ground_truth = [ [ [ref1a], [ref1b], [ref1c] ], ..., [ [refna], [refnb] ] ]

                # ground truth
                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(
                            lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}],
                            img_caps
                        )
                    )  # remove <start> and pads
                    ground_truth.append(img_captions)

                # prediction
                _, preds = torch.max(scores_copy, dim = 2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                prediction.extend(preds)

                assert len(ground_truth) == len(prediction)

            # calc BLEU-4 and CIDEr score
            metrics = Metrics(ground_truth, prediction, self.rev_word_map)
            bleu4 = metrics.belu()[3] # BLEU-4
            cider = metrics.cider() # CIDEr

            print(
                '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}, CIDEr - {cider}\n'.format(
                    loss = losses,
                    top5 = top5accs,
                    bleu = bleu4,
                    cider = cider
                )
            )

        return bleu4


    def run_train(self):
        # epochs
        for epoch in range(self.start_epoch, self.epochs):

            # decay learning rate if there is no improvement for 8 consecutive epochs
            # terminate training if there is no improvement for 20 consecutive epochs
            if self.epochs_since_improvement == 20:
                break
            if self.epochs_since_improvement > 0 and self.epochs_since_improvement % 8 == 0:
                adjust_learning_rate(self.decoder_optimizer, 0.8)
                if self.fine_tune_encoder:
                    adjust_learning_rate(self.encoder_optimizer, 0.8)

            # train an epoch
            self.train(epoch = epoch)

            # validate an epoch
            recent_bleu4 = self.validate()

            # epochs num since last improvement
            is_best = recent_bleu4 > self.best_bleu4
            self.best_bleu4 = max(recent_bleu4, self.best_bleu4)
            if not is_best:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (self.epochs_since_improvement,))
            else:
                self.epochs_since_improvement = 0

            # save checkpoint
            save_checkpoint(
                epoch = epoch,
                epochs_since_improvement = self.epochs_since_improvement,
                encoder = self.encoder,
                decoder = self.decoder,
                encoder_optimizer = self.encoder_optimizer,
                decoder_optimizer = self.decoder_optimizer,
                bleu4 = recent_bleu4,
                is_best = is_best
            )
