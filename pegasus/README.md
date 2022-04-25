# Text summarization number probing
## Pegasus

This folder contains the code to run number probing on Pegasus.

The three basic tasks for number probing (List Maximum, Decoding, Addition) were first proposed for static and contextual embeddings in Wallace et al. (2019). These will be implemented here in a new format that relies on generally available and up-to-date libraries as of 2022.
Beyond these, several additional number probing tests will be supported.

TODO:
1. Add protections against exploding gradients. Changing hyperparameters may be enough in this case, with monitoring for infinity/nan issues.
2. Revisit local/relative import method to prevent future breaks.
3. Test percent, basis points, and units modes.
