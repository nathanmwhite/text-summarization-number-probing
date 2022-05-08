# Text summarization number probing
## Pegasus

This folder contains the code to run number probing on Pegasus.

The three basic tasks for number probing (List Maximum, Decoding, Addition) were first proposed for static and contextual embeddings in Wallace et al. (2019). These will be implemented here in a new format that relies on generally available and up-to-date libraries as of 2022.

Beyond these, several additional number probing tests will be supported, including:
1. Percent Decoding -- Percents in the form of "5 percent", "5.2 percent", or "five percent" are decoded by an architecture parallel to the base Decoding architecture.
2. Basis Points Decoding -- A variant of the Percent Decoding task, basis point expressions such as "5 basis points" or "five basis points" are decoded by an architecture parallel to the base Decoding architecture.
3. Units -- Numerical elements are combined with units such as "kilograms", "employees", and other possible units that numerical elements can quantify, as gleaned from the financial news dataset Malo et al. (2014), with a task seeking to identify the unit as a classification task using an MLP with a classification layer at the top.
4. Units in context -- This is similar to Units in concept, but data containing numerical elements with units from Malo et al. (2014) is combined with the number for which the unit must be identified is used as a second input, with the two separated by a sep_token. The combined sequence is embedded and fed into a BiLSTM layer with a classification layer at the top to identify the correct units for the numerical element.
5. Ranges -- This is somewhat similar to Addition in concept, but data containing numerical ranges with units are inputted into an embedder with a siamese MLP at the top that provides the starting and ending numerals representing the endpoints. Data as input is natural language input including sequences such as "2015-2020", "from 6 to 10", "3 to 5", and so on.
6. Mixed Numeracy -- This is similar to Decoding in concept, but data containing numerical elements with word-based expansion of large numbers is used, with datapoints such as "16.5 mn" or "16.5 million". These are embedded and fed into a MLP for decoding, with the numerical representation matching the real-world number under a logarithmic scale.

Other planned tasks:
1. Layers -- This is similar to Decoding in concept and architecture, except that only one Encoding layer derived from the original embedding model is involved, in order to identify which parts of the model are involved in numeracy representation.

Additional considerations:
1. Changes -- Another possible task is handling changes: "sales increased by 8 % to EUR 155.2 mn" or "operating profit rose to EUR 31.1 mn from EUR 17.1 mn in 2004".

TODO:
1. Add protections against exploding gradients. Changing hyperparameters may be enough in this case, with monitoring for infinity/nan issues.
2. Revisit local/relative import method to prevent future breaks.
3. Test units and context units tasks.
4. Confirm that MSE calculations for Ranges task fits the model architecture.
