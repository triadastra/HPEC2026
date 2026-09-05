import torch

from src.models.gpt import GPTComboAxial


def _make_gpt(dims: int, combo_encoder: str = "embeddings") -> GPTComboAxial:
    coords = [(0, 0, 0), (0, 1, 1)]
    return GPTComboAxial(
        variant=f"cross_attention_{dims}d",
        num_numeric_features=1,
        num_states=1,
        num_commodities=2,
        num_flows=2,
        d_model=8,
        nhead=1,
        num_layers=1,
        mlp_ratio=1.0,
        dropout=0.0,
        attn_dropout=0.0,
        dims=dims,
        num_combos=len(coords),
        features_per_group=1,
        combo_coords=coords,
        lattice_dims=(1, 2, 2),
        combo_encoder=combo_encoder,
        state_embed_dim=2,
        comm_embed_dim=3,
        flow_embed_dim=2,
    )


def test_gpt_combo_encodes_axes_not_promoted_to_attention():
    model_2d = _make_gpt(2)
    assert model_2d.leftover.leftover == [1, 2]
    assert model_2d.leftover.out_dim == 5
    assert model_2d.axial.in_proj.in_features == 6
    assert model_2d.in_proj_raw.in_features == 6

    model_3d = _make_gpt(3, combo_encoder="onehot")
    assert model_3d.leftover.leftover == [2]
    assert model_3d.leftover.out_dim == 2
    assert model_3d.axial.in_proj.in_features == 3

    model_4d = _make_gpt(4)
    assert model_4d.leftover.leftover == []
    assert model_4d.leftover.out_dim == 0
    assert model_4d.axial.in_proj.in_features == 1


def test_gpt_combo_predictions_and_gradients_depend_on_leftover_identity():
    torch.manual_seed(7)
    model = _make_gpt(2)
    x = torch.ones(1, 3, 2, 1)

    prediction = model(x)

    assert prediction.shape == (1, 2, 2)
    assert not torch.allclose(prediction[:, 0], prediction[:, 1])

    prediction.sum().backward()
    for embedding in model.leftover.embeds:
        assert embedding.weight.grad is not None
        assert torch.isfinite(embedding.weight.grad).all()
