.. M2 Coursework documentation master file, created by
   sphinx-quickstart on Thu Mar 28 14:20:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to M2 Coursework's documentation!
=========================================
Executables (src/)
------------------
.. autofunction:: src.ddpm_train
.. autofunction:: src.fashion_mnist_train
.. autofunction:: src.mnist_classifier_train

utils
------------------
.. autoclass:: utils.CNNBlock
    :members:
.. autoclass:: utils.CNN
    :members:
.. autoclass:: utils.CNNClassifier
    :members:
.. autoclass:: utils.DDPM
    :members:
.. autoclass:: utils.FashionMNISTDM
    :members:
.. autoclass:: utils.DiffusionModelTrainer
    :members:

.. autofunction:: utils.ddpm_schedules
.. autofunction:: utils.save_pickle
.. autofunction:: utils.load_pickle
.. autofunction:: utils.create_dir_if_required
.. autofunction:: utils.calc_loss_per_epoch
.. autofunction:: utils.find_latest_model
.. autofunction:: utils.get_feature_vector
.. autofunction:: utils.calculate_fid
.. autofunction:: utils.save_images
