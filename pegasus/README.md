# Text summarization number probing
## Pegasus

This folder contains the code to run number probing on Pegasus.

The three basic tasks for number probing (List Maximum, Decoding, Addition) were first proposed for static and contextual embeddings in Wallace et al. (2019). These will be implemented here in a new format that relies on generally available and up-to-date libraries as of 2022.

Beyond these, several additional number probing tests will be supported, including:
1. Percent Decoding -- Percents in the form of "5 percent", "5.2 percent", or "five percent" are decoded by an architecture parallel to the base Decoding architecture.
2. Basis Points Decoding -- A variant of the Percent Decoding task, basis point expressions such as "5 basis points" or "five basis points" are decoded by an architecture parallel to the base Decoding architecture.
3. Units -- Numerical elements are combined with units such as "kilograms", "employees", and other possible units that numerical elements can quantify, as gleaned from the financial news dataset Malo et al. (2014), with a task seeking to identify the unit as a classification task using an MLP with a classification layer at the top.

Other planned tasks:
1. Units in context -- This is similar to units in concept, but data containing numerical elements with units from Malo et al. (2014) is used as one embedded input sequence, and the number for which the unit must be identified is used as a second embedded input, both of which are fed into a BiLSTM or other architecture with a classification layer at the top to identify the correct units for the numerical element.
2. TBA

TODO:
1. Add protections against exploding gradients. Changing hyperparameters may be enough in this case, with monitoring for infinity/nan issues.
2. Revisit local/relative import method to prevent future breaks.
3. Test percent, basis points, and units modes.
