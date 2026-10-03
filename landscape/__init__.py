"""landscape — the superconformal-index pipeline of the 4d N=1 landscape program
(arXiv:2408.02953) for a product of simple gauge groups.

Modules, in dependency order: lie (Lie data from highest weights), model (theories,
records, canonical form, flavor lattice, index interfaces, the identity of a fixed point),
notation (the index-string notation), convert (legacy formats -> model), store (character
store keyed by Dynkin label), singlet (product singlet projection and the call of the native
expansion engine), form (the series of a theory), index (the index engine), amax (a-maximization and validators), post (operator
extraction and consistency conditions), record (one theory -> one record), enumerate (the
next level and the duplicate rule), driver (a seed level by level); schema/ holds the JSON
schemas of theories and records.  The native extension `landscape_native` (Rust) is required: it
expands the series, projects onto gauge singlets and computes the characters."""
