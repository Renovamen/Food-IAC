"""
Mostly inspired from Artexte's code: https://github.com/artetxem/undreamt
"""
import torch
import torch.nn as nn

class GlobalAttention(nn.Module):
    def __init__(self, dim: int, alignment_function: str = 'general') -> None:
        super(GlobalAttention, self).__init__()

        self.alignment_function = alignment_function
        if self.alignment_function == 'general':
            self.linear_align = nn.Linear(dim, dim, bias=False)
        elif self.alignment_function != 'dot':
            raise ValueError('Invalid alignment function: {0}'.format(alignment_function))

        self.softmax = nn.Softmax(dim=1)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(
        self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): batch*dim
            context (torch.Tensor): length*batch*dim
            mask (torch.Tensor): batch*dim
        """

        context_t = context.transpose(0, 1)  # batch*length*dim

        # Compute alignment scores
        q = query if self.alignment_function == 'dot' else self.linear_align(query)
        align = context_t.bmm(q.unsqueeze(2)).squeeze(2)  # batch*length

        # Mask alignment scores
        if mask is not None:
            align.data.masked_fill_(mask.bool(), -float('inf'))

        # Compute attention from alignment scores
        attention = self.softmax(align)  # batch*length

        # Computed weighted context
        weighted_context = attention.unsqueeze(1).bmm(context_t).squeeze(1)  # batch*dim

        # Combine context and query
        return self.tanh(self.linear_context(weighted_context) + self.linear_query(query))
