# Food Image Aesthetic Captioning Dataset

This folder contains the captions related to each of the images stored in `json` files. The format follows the [Karpathy's split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip):

```js
{
  "images": [
    {
      "sentences": [
        {
          "raw": "the first caption",
          "tokens": ["the", "first", "caption"]
        },
        {
          "raw": "the second caption",
          "tokens": ["the", "second", "caption"]
        }
      ],
      "filename": "id-of-the-image.jpg",
      "url": "url-of-the-image",
      "split": "train"  // train / val / test
    },
    ...
  ]
}
```

train : val : test ≈ 6 : 1 : 1