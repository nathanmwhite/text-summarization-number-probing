# Probing of Quantitative Values in Abstractive Summarization Models

This repository contains the code related to the upcoming paper, "Probing of Quantitative Values in Abstractive Summarization Models".

This work seeks to probe the ability of abstractive summarization models to represent quantitative values as typically found in financial news in terms of absolute magnitude, relative order, and the units represented by the quantitative values. 

Given the prevalence of number hallucination in abstractive summarization, where numerical values not present in the summarization input appear in the output summary, a methodology needs to be developed to evaluate the degree to which an encoder's modeling of the values in the input is successful and recoverable to best determine which abstractive summarization model architecture is safest in a financial context where accurate representation of quantitative values is critical.

This represents one step of the larger Hmong Medical Text Summarization project, which seeks to develop an artificial intelligence-based system that combines abstractive text summarization and machine translation tasks for a low-resource language.

This repository contains the following folders:
1. pegasus : code to run number probing for each task, with Pegasus defined as the default.
2. s2s_ft : code for UniLMv2 derived from microsoft/unilm/s2s-ft/s2s_ft and edited for compatibility.
3. units_processing : processing code and text files containing the units for the Units and Context_Units probing tasks.
