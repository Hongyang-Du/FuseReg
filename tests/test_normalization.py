"""The baseline checkpoint must not silently inherit the oldnorm convention."""
import pytest

from eval_reconstruction import resolve_output_normalization


def test_dino_no_drop_requires_explicit_normalization():
    with pytest.raises(ValueError, match='normalization is unresolved'):
        resolve_output_normalization('dinov3mls-vit-l16', 0.0)


@pytest.mark.parametrize('convention', ['raw', 'encoder'])
def test_explicit_baseline_convention_is_preserved(convention):
    assert resolve_output_normalization('dinov3mls-vit-l16', 0.0, convention) == convention


@pytest.mark.parametrize('encoder,p_drop', [('dinov3mls-vit-l16', .95), ('dinov3mls-vit-l16', .1)])
def test_sourced_oldnorm_lineages_keep_their_default(encoder, p_drop):
    assert resolve_output_normalization(encoder, p_drop) == 'encoder'
