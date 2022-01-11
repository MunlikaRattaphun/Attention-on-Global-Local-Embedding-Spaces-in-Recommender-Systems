# Attention on global-local representation spaces in recommender systems

This is our official implementation for the paper:

Munlika Rattaphun, Wen-Chieh Fang, and Chih-Yi Chiu, "[Attention on global-local representation spaces in recommender systems](https://ieeexplore.ieee.org/abstract/document/9638260)," IEEE Transactions on Computational Social Systems (TCSS), 2021.

In this study, we present a novel clustering-based collaborative filtering (CF) method for recommender systems. Clustering-based CF methods can effectively deal with data sparsity and scalability problems. However, most of them are applied to a single representation space, which might not characterize complex user-item interactions well. We argue that the user-item interactions should be observed from multiple views and characterized in an adaptive way. To address this issue, we leveraged the global and local properties to construct multiple representation spaces by learning various training datasets and loss functions. An attention network was built to generate a blended representation according to the relative importance of the representation spaces for each user-item pair, providing a flexible way to characterize diverse user-item interactions. Substantial experiments were evaluated on four popular benchmark datasets. The results show that the proposed method is superior to several CF methods where only one representation space is considered.

## Citation

    @article{rattaphun2021attention,
       title={Attention on Global-Local Representation Spaces in Recommender Systems},
       author={Rattaphun, Munlika and Fang, Wen-Chieh and Chiu, Chih-Yi},
       journal={IEEE Transactions on Computational Social Systems},
       year={2021},
       publisher={IEEE}
    }
