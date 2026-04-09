#include <data_structure/registration/hash_table_utils.h>
#include <string.h>

void hash_result_to_string(hash_result_t result, char *string){
    switch (result)
    {
    case HASH_ENTRY_FOUND:
        strcpy(string, "HASH_ENTRY_FOUND");
        break;
    case HASH_ENTRY_ADDED:
        strcpy(string, "HASH_ENTRY_ADDED");
        break;
    case HASH_ENTRY_NOT_FOUND:
        strcpy(string, "HASH_ENTRY_NOT_FOUND");
        break;
    case HASH_ENTRY_ALREADY_EXISTS:
        strcpy(string, "HASH_ENTRY_ALREADY_EXISTS");
        break;
    case HASH_ENTRY_UNKOWN_DATATYPE:
        strcpy(string, "HASH_ENTRY_UNKOWN_DATATYPE");
        break;
    default:
        strcpy(string, "UNKNOWN CODE");
        break;
    }
}
