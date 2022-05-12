import torch
import torch.nn as nn
import numpy as np
import pre_data
import transformers
from transformers import BertTokenizer,BertTokenizerFast

s="Title:Nanostructured Tungstate-Derived Copper for Hydrogen Evolution Reaction and Electroreduction of CO2 in Sodium Hydroxide Solutions\nAbstract:Electroreduction of CO2 became an important topic recently because it can reduce the atmospheric CO2 levels and simultaneously synthesize chemical fuels. However, efficient conversion of CO2 to produce fuels remains a challenge because a proper electrocatalyst is needed to make this CO2 reduction process more selective and efficient. In this study, we prepared nanostructured tungstate-derived copper to test its application in CO2 reduction. The prepared copper tungstate (CuWO4) nanomaterials were first characterized by analytical techniques such as transmission electron microscopy, X-ray diffraction, and X-ray photoelectron spectroscopy to determine the particle size, crystallinity, purity, and composition. Then, the CuWO4 nanomaterials were further investigated in an aqueous solution containing 0.1 M NaOH by electrochemical cyclic voltammetry (CV) and linear sweep voltammetry (LSV) techniques. The CO2 electroreduction experiments were carried out in 0.1 M NaOH with the presence of CO2, and the analysis of electrochemical results shows that nanostructured CuWO4 performs better in comparison with CuO-a well-known electrocatalyst for reducing CO, to nongaseous carbon-containing products such as alcohols-because of poisoning effects of adsorbed CO, or its adsorbed-reduced intermediates on hydrogen evolution reaction. Our results also show that CO2-reduction intermediates adsorbed strongly on the surface of CuWO4, which increases the overpotential for hydrogen evolution reaction on the surface of CuWO 4 by as much as 230 mV against the 70 mV for CuO, at a current density of 0.8 mA cm(-2).\nDoi:10.1021_acs.jpcc.9b07133"
tokenizer = BertTokenizerFast.from_pretrained("recobo/chemical-bert-uncased")
t=tokenizer(s,truncation=True,padding=True,return_offsets_mapping=True,max_length=512,return_tensors="pt")
print(t.offset_mapping)
t1=tokenizer.tokenize(s)
print(t1)
