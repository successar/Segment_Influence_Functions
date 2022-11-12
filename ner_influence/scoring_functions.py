from typing import Callable, Iterable

import numpy as np
from scipy.stats import entropy

from ner_influence.entity_influence_indexing import EntityIndexer, get_token_gradients
from ner_influence.instance_influence_indexing import (InstanceIndexer,
                                                       get_instance_gradient)
from ner_influence.scaffolding import Instance

Aggregator = Callable[[Iterable[float]], float]

def random_scorer(instances: Iterable[Instance]) -> Iterable[tuple[str, float]]:
    for instance in instances:
        yield instance["id"], np.random.rand()


def instance_loss_scorer(instances: Iterable[Instance]) -> Iterable[tuple[str, float]]:
    for instance in instances:
        yield instance["id"], instance["loss"]


def instance_gradient_scorer(instances: Iterable[Instance]) -> Iterable[tuple[str, float]]:
    for instance in instances:
        gradient_norm = np.linalg.norm(get_instance_gradient(instance, normalize=False)[0])
        yield instance["id"], gradient_norm


def token_conditional_loss_scorer(
    instances: Iterable[Instance], aggregator: Aggregator
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        cond_probs = instance["conditional_probs"](labels=instance["gold_labels"])
        token_losses = cond_probs[np.arange(cond_probs.shape[0]), instance["gold_labels"]]
        token_losses = -np.log(token_losses)
        yield instance["id"], aggregator(token_losses)


def token_marginal_loss_scorer(
    instances: Iterable[Instance], aggregator: Aggregator
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        marg_probs = instance["marginal_probs"]
        token_losses = marg_probs[np.arange(marg_probs.shape[0]), instance["gold_labels"]]
        token_losses = -np.log(token_losses)
        yield instance["id"], aggregator(token_losses)


def token_conditional_entropy_scorer(
    instances: Iterable[Instance], aggregator: Aggregator
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        cond_probs = instance["conditional_probs"](labels=instance["gold_labels"])
        yield instance["id"], aggregator(entropy(cond_probs, axis=-1))


def token_marginal_entropy_scorer(
    instances: Iterable[Instance], aggregator: Aggregator
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        marginal_probs = instance["marginal_probs"]
        yield instance["id"], aggregator(entropy(marginal_probs, axis=-1))


def token_gradient_scorer(
    instances: Iterable[Instance], aggregator: Aggregator
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        gradient = np.linalg.norm(get_token_gradients(instance, normalize=False), axis=-1)
        assert gradient.shape == (len(instance["tokens"]),)
        yield instance["id"], aggregator(gradient)

################################################################################
#                        Influence based scorers                               #
################################################################################

def instance_influence_scorer(
    instances: Iterable[Instance], indexer: InstanceIndexer
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        yield instance["id"], instance["influence"]

def token_influence_scorer(
    instances: Iterable[Instance], aggregator: Aggregator, indexer: EntityIndexer
) -> Iterable[tuple[str, float]]:
    for instance in instances:
        yield instance["id"], instance["entity_influence"]