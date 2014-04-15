def init_hash_structure(type_ht, hash_table_size, bucket_vector_length):
    uh = uhash()
    uh.hash_table_size = hash_table_size
    uh.type_ht = type_ht
    uh.hashed_data_length = bucket_vector_length
    uh.prime = const.prime_default
    if type_ht == const.ht_linked_list:
        uh.hash_table.ll_hash_table = []
    elif type_ht == const.ht_hybrid_chains:
        #something
    uh.main_hash_a = np.random.random_integers(1, const.max_hash_rnd, uh.hashed_data_length)
    uh.control_hash = np.random.random_integers(1, const.max_hash_rnd, uh.hashed_data_length)
    return uh
