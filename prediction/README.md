## Pedestrian Trajectory Prediction Models

This repository contains a robust dataset tailored for deep learning applications in Pedestrian Trajectory Prediction. In the field of autonomous driving and crowd behavior analysis, predicting the future trajectories of pedestrians is a key component. Here, we introduce three state-of-the-art models - Social LSTM, Y-Net, and NSP, which are employed to offer baseline of trajectory predictions.



### Social LSTM
The Social LSTM model takes into account the social dynamics among pedestrians. It allows the model to not only understand individual behaviors but also capture the interactions within a group of pedestrians. This LSTM-based model leverages a social pooling layer to share information across multiple LSTM units, making it possible to anticipate complex pedestrian behaviors in crowded scenes.


### Y-Net
Human trajectory forecasting is a complex problem with inherent uncertainty. This uncertainty arises from both known and unknown sources, such as long-term goals and the intent of other agents. Y-net proposes separating this uncertainty into epistemic uncertainty, related to long-term goals, and aleatoric uncertainty, related to waypoints and paths. This approach introduces multimodality in goal predictions and path variations to capture these uncertainties. Y-net also introduces a new long-term trajectory forecasting scenario with prediction horizons up to a minute, surpassing previous works.


### NSP
Neural Social Physics (NSP) is proposed for trajectory prediction, which combines both model-based and model-free methodologies. The approach integrates an explicit physics model with learnable parameters into a deep neural network. The physics model acts as a significant inductive bias in modeling pedestrian behaviors, and the rest of the network offers robust data-fitting capabilities for system parameter estimation and dynamics stochasticity modeling.




## Benchmarks
We provide benchmarks and pretrained models for Pedestrian Trajectory Prediction.


### Pedestrian Trajectory Prediction
|**Name**|**Map**|**ADE<sub>5<sub> &darr;**|**FDE<sub>5<sub> &darr;**| **ADE<sub>20<sub> &darr;** | **FDE<sub>20<sub> &darr;** | **Pretrained** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**Vanilla LSTM**| X|1.156 | 2.205 | 1.601 | 3.157 | <a href="">TBD</a>|
|**Social-LSTM**| X | 1.336 | 2.554 | 1.319 | 2.519 | <a href="">TBD</a>|
|**Y-NET**| X | 1.188 | 2.427 | 0.640 | 1.547 | <a href="">TBD</a>|
|**Y-NET**| O | 1.036 | 2.306 | 0.596 | 1.370 | <a href="">TBD</a>|
|**NSP-SFM**| X | 1.036 | 1.947 | 0.529 | 0.936 | <a href="">TBD</a>|
|**NSP-SFM**| O | 0.808 | 1.549 | 0.443 | 0.807 | <a href="">TBD</a>|
