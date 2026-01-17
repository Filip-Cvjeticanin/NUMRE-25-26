import numpy as np
from PyModuli.SimpleTest import train_on_N_examples, load_and_test
from PyModuli.SimpleTest import evaluate_model_on_N_examples

if __name__ == "__main__":

    #
    #
    #              NAPOMENE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    #

    #                1. Za pokretanje koda u svim odjeljcima osim prvog. Potrebno je skinuti SkupPodataka.csv
    #                   sa MEGA download linka danog u ./Podatci direktoriju i staviti datoteku u isti direktorij.
    #
    #                2. Za pokretanje trećeg odjeljka potrebno je prvo pokrenuti drugi.
    #
    #                3. Za pokretanje četvrtog odjeljka potrebno je skinuti model pod nazivom model-seq-3.pt sa MEGA
    #                   download linka danog u ./Modeli direktoriju i staviti datoteku u isti direktorij.


    DATASET = "./Podatci/SkupPodataka.csv"
    MODELPATH_TEST = "./Modeli/model-seq-3.pt"
    MODELPATH_TRENING = "./Modeli/modelProba.pt"

    #Poziv za treniranje novog modela
    #=============================================================================
    #train_on_N_examples(N=1,
    #                    test_examples_num= 0,
    #                    dataset_filepath= DATASET,
    #                    save_path = MODELPATH_TRENING,
    #                    epoch_num=1,
    #                    example_by_example=True)

    #Poziv za testiranje vec treniranog modela i testiranje na nekim primjerima
    #NAPOMENA: Potrebno je prvo istrenirati model kodom u odjeljku iznad.
    #=============================================================================
    #load_and_test(load_path=MODELPATH_TRENING,
    #              dataset_filepath= DATASET,
    #              test_example_start_idx= 10,
    #              test_example_num= 1,
    #              summary_sentences_num= 6)

    # Poziv za testiranje vec treniranog modela i testiranje na nekim primjerima
    # NAPOMENA: Za ovaj kod potrebno je skinuti model-seq-3.pt sa MEGA
    #             download linka danog u ./Modeli direktoriju i staviti datoteku u isti direktorij.
    # =============================================================================
    #load_and_test(load_path=MODELPATH_TEST,
    #              dataset_filepath= DATASET,
    #              test_example_start_idx= 3000,
    #              test_example_num= 5,
    #              summary_sentences_num= 6)

    # evaluate_model_on_N_examples(
    #     load_path=MODELPATH_TEST,
    #     dataset_filepath=DATASET,
    #     start_idx=200,
    #     num_examples=1,
    #     match_abstract=True,
    #     fixed_ratio=0.15,
    # )


    pass




