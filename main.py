import numpy as np
from PyModuli.TestModule1 import hello_from_test_module
from PyModuli.SimpleTest import experimental_train_and_test
from PyModuli.SimpleTest import train_on_N_examples, load_and_test


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


    # Poziv za jednostavan test summary.
    # ======================================
    #experimental_train_and_test()



    # Poziv za treniranje nocog modela na 10 primjera
    # =============================================================================
    #train_on_N_examples(N=10,
    #                    test_examples_num= 2,
    #                    dataset_filepath= "../Podatci/SkupPodataka.csv",
    #                    save_path = "../Modeli/model-test.pt",
    #                    epoch_num=10,
    #                    example_by_example=True)



    # Poziv za testiranje vec treniranog modela i testiranje na nekim primjerima
    # NAPOMENA: Potrebno je prvo istrenirati model kodom u odjeljku iznad.
    # =============================================================================
    #load_and_test(load_path="./Modeli/model-test.pt",
    #              dataset_filepath= "./Podatci/SkupPodataka.csv",
    #              test_example_start_idx= 10,
    #              test_example_num= 2,
    #              summary_sentences_num= 6)



    # Poziv za testiranje vec treniranog modela sa mega download linka: (model-seq-3.pt)
    # =============================================================================
    #load_and_test(load_path="./Modeli/model-seq-3.pt",
    #              dataset_filepath= "./Podatci/SkupPodataka.csv",
    #              test_example_start_idx= 200,
    #              test_example_num= 5,
    #              summary_sentences_num= 6)


    pass