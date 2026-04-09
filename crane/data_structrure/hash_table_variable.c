#include <data_structure/registration/hash_table_variable.h>

void create_variable_table(variable_table_t *variable_table)
{
    variable_table->root = NULL;
    mtx_init(&variable_table->mutex, mtx_plain);
}


hash_result_t find_variable_entry(variable_entry_hh_t **table,
    semantic_id_t *sid)
{
    variable_entry_hh_t tmp;
    variable_entry_hh_t *entry_in_the_table = NULL;

    memset(&tmp, 0, sizeof(variable_entry_hh_t));
    memcpy((void *)&tmp.sid, (void *)sid, sizeof(semantic_id_t));
    HASH_FIND(hh_sid, *table, &tmp.sid, sizeof(tmp.sid), entry_in_the_table);
    if (entry_in_the_table)
    {
        return HASH_ENTRY_FOUND;
    }
    else
    {
        return HASH_ENTRY_NOT_FOUND;
    }
}

hash_result_t get_variable_entry(variable_entry_hh_t **table,
    semantic_id_t *sid, variable_entry_t **entry)
{
    variable_entry_hh_t tmp;
    variable_entry_hh_t *entry_in_the_table = NULL;

    memset(&tmp, 0, sizeof(variable_entry_hh_t));
    memcpy((void *)&tmp.sid, (void *)sid, sizeof(semantic_id_t));
    HASH_FIND(hh_sid, *table, &tmp.sid, sizeof(tmp.sid), entry_in_the_table);
    if (entry_in_the_table)
    {
        *entry = &entry_in_the_table->entry;
        return HASH_ENTRY_FOUND;
    }
    else
    {
        return HASH_ENTRY_NOT_FOUND;
    }
}

