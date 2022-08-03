# Probing of Quantitative Values in Abstractive Summarization Models
## text-summarization-number-probing/pegasus

This folder contains the code to run quantitative value probing tasks, with Pegasus defined as the default model type. Other model types available if explicitly specified include:
1. T5
2. T5+SSR
3. BART
4. DistilBART
5. ProphetNet

Several number probing tasks are supported, including:
1. Addition -- Adding of two float values. This is adapted conceptually from the integer version in Wallace et al. (2019), with a fully new implementation that relies on generally available and up-to-date libraries as of 2022.
2. Percent Decoding -- Percents in the form of "5.2%" are decoded by an architecture parallel to the base Decoding architecture.
3. Basis Points Decoding -- A variant of the Percent Decoding task, basis point expressions are decoded by an architecture parallel to the base Decoding architecture.
4. Units -- Numerical elements are combined with units such as "kilograms", "employees", and other possible units that numerical elements can quantify, as gleaned from the financial news dataset Malo et al. (2014), with a task seeking to identify the unit as a classification task using an MLP with a classification layer at the top.
5. Ranges -- This is somewhat similar to Addition in concept, but data containing numerical ranges with units are inputted into an embedder with a siamese MLP at the top that provides the starting and ending numerals representing the endpoints. Data as input is natural language input including sequences such as "2015-2020", "from 6 to 10", "3 to 5", and so on.
6. Orders -- This is similar to Decoding in concept, but data containing numerical elements with word-based expansion of large numbers is used, with datapoints such as "16.5 mn" or "16.5 million". These are embedded and fed into a MLP for decoding, with the numerical representation matching the real-world number under a logarithmic scale.

Other tasks that are technically supported but not considered at this time for ongoing research include:
1. Units in Context -- This is similar to Units in concept, but data containing numerical elements with units from Malo et al. (2014) is combined with the number for which the unit must be identified is used as a second input, with the two separated by a sep_token. The combined sequence is embedded and fed into a BiLSTM layer with a classification layer at the top to identify the correct units for the numerical element. While this is technically implemented in the existing code, the units in context task may involve proprietary data, and is therefore left for future work.
2. List Maximum -- First proposed in Wallace et al. (2019), this seeks which value is the largest from a set of five integer or float values.
3. Integer Decoding -- First proposed in Wallace et al. (2019), this seeks to have the model reproduce as a float value the original integer embedded in the input.

Other tasks that should be considered in a later stage:
1. Layers -- This is similar to Decoding in concept and architecture, except that only one Encoding layer derived from the original embedding model is involved, in order to identify which parts of the model are involved in numeracy representation.
2. Changes -- Another consideration is handling changes: "sales increased by 8 % to EUR 155.2 mn" or "operating profit rose to EUR 31.1 mn from EUR 17.1 mn in 2004".
