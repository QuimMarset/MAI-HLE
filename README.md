# MAI-HLE

## Master in Artificial Intelligence - Human Language Engineering - UPC

### Final Project - Relation Extraction

This project implements several models to solve the SemEval-2010 Task 8.

The task features a dataset where the goal is to find the relation class between two entities in a sentence.
For example, we might have the sentence "The <e1>pollution</e1> was caused by the <e2>shipwreck</e2>", where the entities are "pollution" and "shipwreck", and the goal is finding the relation class "Cause-Effect(e2, e1)", as "shipwreck" is the cause for 
the bad effect of "pollution".

We implemented different classification models, from old ones based on RNNs to using the latest Transformer architecture like BERT.

We include all the code plus the final report to have a good understanding of all the processes we have followed.
