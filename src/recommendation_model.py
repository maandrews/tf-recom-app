"""
Class for a complete recommendation model
"""

import tensorflow_recommenders as tfrs
from item_model import CandidateModel
from customer_model import QueryModel


class RecomModel(tfrs.models.Model):
    def __init__(self, unique_ids, unique_items, unique_items_for_metrics,
                 layer_sizes=None, query_model=None, candidate_model=None):
        super().__init__()
        if query_model is None:
            self.query_model = QueryModel(layer_sizes, unique_ids)
        else:
            self.query_model = query_model

        if candidate_model is None:
            self.candidate_model = CandidateModel(layer_sizes, unique_items)
        else:
            self.candidate_model = candidate_model

        self.task = tfrs.tasks.Retrieval(
            metrics = tfrs.metrics.FactorizedTopK(
                candidates = unique_items_for_metrics.map(self.candidate_model),),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            'USER_ID': features['USER_ID'],
        })

        item_embeddings = self.candidate_model({
            'ITEM': features['ITEM'],
        })

        return self.task(query_embeddings, item_embeddings, compute_metrics=not training)

    def get_query_embedding(self, features):
        return self.query_model({
            'USER_ID': features['USER_ID'],
        })
