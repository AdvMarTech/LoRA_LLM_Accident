<h1 align="center">
Intelligent Analysis of Ship Collision Accidents via Low-Rank<br>
Adaptation-Based Fine-Tuning of Medium-Scale Large Language Models
</h1>

<p align="center">
Jun Ma <sup>1,2, iD</sup>&nbsp;&nbsp;&nbsp;
Liang Cao <sup>3,*, iD</sup>&nbsp;&nbsp;&nbsp;
Yinwei Feng <sup>4, iD</sup>&nbsp;&nbsp;&nbsp;
Çağlar Karatuğ <sup>5</sup><br>
Muge Buber <sup>6, iD</sup>&nbsp;&nbsp;&nbsp;
Xinjian Wang <sup>7,8,*, iD</sup>
</p>

<p>
<sup>1</sup> School of Computer Science, Xijing University, Xi’an, 710123, P. R. China<br>
<sup>2</sup> Xi’an Key Laboratory of Human-Machine Integration and Control Technology for Intelligent Rehabilitation, Xijing University, Xi’an, 710123, P. R. China<br>
<sup>3</sup> Naval Architecture and Shipping College, Guangdong Ocean University, Zhanjiang 524088, P. R. China<br>
<sup>4</sup> Department of Logistics and Maritime Studies, The Hong Kong Polytechnic University, Hong Kong, 999077, P. R. China<br>
<sup>5</sup> Maritime Faculty, Istanbul Technical University, Tuzla 34940, Istanbul, Turkey<br>
<sup>6</sup> Maritime Faculty, Dokuz Eylul University, Izmir 35390, Turkey<br>
<sup>7</sup> Navigation College, Dalian Maritime University, Dalian 116026, P. R. China<br>
<sup>8</sup> Liverpool Logistics, Offshore and Marine (LOOM) Research Institute, Liverpool John Moores University, Liverpool L3 3AF, UK
</p>

<p>
* Corresponding author: caoliang@gdou.edu.cn, X.Wang1@ljmu.ac.uk
</p>

---

## Abstract

The rapid advancement of intelligent maritime accident analysis requires processing large-scale, multilingual data across wide geographic regions. However, significant challenges remain in objectively constructing Risk Influencing Factors (RIFs) and ensuring accurate information extraction with limited computational resources. To address these gaps, a framework for intelligent analysis of ship collision accidents based on Low-Rank Adaptation (LoRA) fine-tuning of medium-scale large language models (LLMs) with limited labeled data was proposed. A bilingual dataset comprising 503 ship collision accident reports was established, and the RIF ontology was derived using a Grounded Theory approach. Using 60 labeled samples, models with ≤ 8B parameters were fine-tuned, achieving an F1 score of 94.11% on the most challenging accident RIF extraction subtask, surpassing base models by 34.82%. Then, the extracted information was transformed into a 1061-row × 24-column training data matrix via a semantic similarity model, enabling construction of a TAN-BN model. Finally, sensitivity analysis was conducted to identify key RIFs, and case studies were performed to evaluate model performance and validate the proposed framework. The research results showed that the proposed approach advances large-scale, cross-lingual intelligent maritime accident report analysis by improving accuracy and efficiency, reducing computational costs, and supporting reliable safety management decisions.

**Keywords:** Maritime safety, Marine accident, Accident analysis, Large Language Models, Low-Rank Adaptation, TAN-BN
