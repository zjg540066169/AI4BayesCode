## vi_block (abstract base -- NOT used directly)

Abstract base for the VI family. Derives from `block_sampler` and
overrides `engine_kind() == engine_kind_t::VI`. Two concrete
subclasses ship in v1: `mean_field_gaussian_vi_block` and
`full_rank_gaussian_vi_block`. Users don't construct `vi_block`
directly.
