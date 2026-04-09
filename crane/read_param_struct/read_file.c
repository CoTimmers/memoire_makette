/* 
    Author: Boris Deroo 
    KU Leuven, Department of Mechanical Engineering, ROB group 
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "read_file/read_file.h"

#if __has_include (<cjson/cJSON.h>)
    #include<cjson/cJSON.h>
#else
    #include<cJSON.h>
#endif

#include <libxml/xmlmemory.h>
#include <libxml/parser.h>

#ifndef NAN
#define NAN 0.0/0.0
#endif

/* CSV file reading */
/* Opens CSV file */
int open_csv_file(FILE** fp, const char *path_to_file)
{
    // Read a reference trajectory from csv file into an array
    *fp = fopen(path_to_file, "r");
    if (*fp == NULL)
    {
        printf("Error opening file: '%s'\n", path_to_file);
        return 0;
    }
    else 
        {printf("Opened file: '%s'\n",path_to_file);}
    return 1;
}

/* 
    reads line from csv file and puts values separated by commas 
    in subsequent double buffers provided in the argument
    ATTENTION: This function can write to pointers that are out of scope!
*/ 
int read_line_from_csv_file(FILE* fp, double *value)
{
    char line[2550];
    if(fgets (line, 2550, fp)!=NULL ) 
    {   
        char* csv_element;
        csv_element = strtok(line, ",");

        while(csv_element != NULL)
        {
            *value = strtod(csv_element,NULL);
            csv_element = strtok(NULL, ",");    
            value++;
        }
            
        return 1;
    }
    else
    {
        printf("Finished reading file\n");
        fclose(fp);
        return 0;
    }
}
void close_csv_file(FILE* fp)
{fclose(fp);}

/* JSON file reading */
int read_from_json_file(const char *filename, param_array_t param_array[], int length_array)
{
    int return_value = CONFIGURATION_FROM_FILE_SUCCEEDED; // Start with succes, changes if error occurs

    /* Open file */
    FILE *infile = fopen (filename, "r");
    if (infile == NULL)
    {
        printf("ERROR: Could not open '%s' input file\n", filename);
        return CONFIGURATION_FROM_FILE_FAILED;

        return 0;
    }

    char buffer[2048];
    fread(buffer, 1, sizeof buffer - 1, infile);
    fclose (infile);
    cJSON *root = cJSON_Parse(buffer);
    if (root == NULL)
    {
        printf("ERROR: Could not interpret '%s'!\n", filename);
        cJSON_Delete(root);
        return_value = CONFIGURATION_FROM_FILE_FAILED;

        return 0;
    }
    
    cJSON *element;
    int i = 0;
    for(i; i < length_array;i++) 
    {
        char foo[strlen(param_array[i].param_name)+1];
        strcpy(foo,param_array[i].param_name);
        char* ch_pos = strchr(foo,'/');
        char subelement[sizeof(foo)];
        memset(subelement,0,sizeof(foo));
        element = root;

        while (ch_pos!=NULL)
        {
            /* GO DEEPA */
            memcpy(subelement, foo, ch_pos-foo);
            element = cJSON_GetObjectItemCaseSensitive(element, subelement);

            int len =  ch_pos-foo+1;
            memmove(foo, foo +len, sizeof(foo)-len);
            memset(subelement,0,sizeof(foo));
            ch_pos = strchr(foo,'/');
        }
        memcpy(subelement, foo, sizeof(foo));
        element = cJSON_GetObjectItemCaseSensitive(element, subelement);

        if (element == NULL)
        {
            if (param_array[i].param_optional > 0)
            {
                printf("Warning: Optional parameter '%s' was not included in input file: '%s'!\n", param_array[i].param_name, filename);
            }
            else
            {
                printf("ERROR: Could not read mandatory parameter '%s' from input file: '%s'!\n", param_array[i].param_name, filename);
                return_value = CONFIGURATION_FROM_FILE_FAILED;
            }
        }
        else
        {
            if(param_array[i].param_type == PARAM_TYPE_INT)
                {*((int*) param_array[i].param_pointer) = element->valueint;}
            else if (param_array[i].param_type == PARAM_TYPE_FLOAT)
                {*((float*) param_array[i].param_pointer) = element->valuedouble;}
            else if (param_array[i].param_type == PARAM_TYPE_DOUBLE)
                {*((double*) param_array[i].param_pointer) = element->valuedouble;}
            else if (param_array[i].param_type == PARAM_TYPE_CHAR)
                {strcpy((char*) param_array[i].param_pointer, (const char*) element->valuestring);}
            else
            {
                printf("ERROR: Parameter %s does not have a valid parameter type specified!\n", param_array[i].param_name);
                return_value = CONFIGURATION_FROM_FILE_FAILED;
            }
        }
    }

    cJSON_Delete(root);
    return return_value;
}

int read_from_XML_file(const char *filename, param_array_t param_array[], int length_array)
{
    int return_value = CONFIGURATION_FROM_FILE_SUCCEEDED; // Start with succes, changes if error occurs
	
    /* Open file */
    xmlDocPtr infile = xmlParseFile(filename);
    if (infile == NULL)
    {
        printf("ERROR: Could not open '%s' input file\n", filename);
        return CONFIGURATION_FROM_FILE_FAILED;
    }

	xmlNodePtr root = xmlDocGetRootElement(infile);

    if (root == NULL)
    {
        printf("ERROR: Could not interpret '%s'!\n", filename);
        xmlFreeDoc(infile);
        return_value = CONFIGURATION_FROM_FILE_FAILED;
    }

    for(int i = 0; i < length_array;i++) 
    {
        char foo[strlen(param_array[i].param_name)+1];
        strcpy(foo,param_array[i].param_name);
        char* ch_pos = strchr(foo,'/');
        char subelement[sizeof(foo)];
        memset(subelement,0,sizeof(foo));
        xmlNodePtr element = root;

        int found_element = 0;  // Looking for element
        element = element->xmlChildrenNode; 

        /* Go down the path */
        while( (ch_pos!=NULL) && (element != NULL) )
        {
            memcpy(subelement, foo, ch_pos-foo);

            /* GO DEEPA */
            /* Loop through elements until found or no more elements */
            while( (found_element == 0) )
            {
                if(element == NULL)
                    {break;} 
                if(!xmlStrcmp(element->name, (const xmlChar*) subelement))  
                {
                    element = element->xmlChildrenNode;
                    found_element = 1;  // Element found
                }
                else
                    {element = element->next;}
            }
            found_element = 0;  // Reset for next iteration 

            int len =  ch_pos-foo+1;
            memmove(foo, foo +len, sizeof(foo)-len);
            memset(subelement,0,sizeof(foo));
            ch_pos = strchr(foo,'/');
        }

        memcpy(subelement, foo, sizeof(foo));
        
        /* Loop through elements until found or no more elements */
        while( (found_element == 0) )
        {
            if(element == NULL)
                {break;} 
            if(!xmlStrcmp(element->name, (const xmlChar*) subelement))     // found end character
                {found_element = 1;}  // Element found
            else
                {element = element->next;}
        }

        if (element == NULL)
        {
            if (param_array[i].param_optional > 0)
                {printf("Warning: Optional parameter '%s' was not included in input file: '%s'!\n", param_array[i].param_name, filename);}
            else
            {
                printf("ERROR: Could not read mandatory parameter '%s' from input file: '%s'!\n", param_array[i].param_name, filename);
                return_value = CONFIGURATION_FROM_FILE_FAILED;
            }
        }
        else
        {
            xmlChar* key = xmlNodeListGetString(infile, element->xmlChildrenNode, 1);

            if(param_array[i].param_type == PARAM_TYPE_INT)
                {*((int*) param_array[i].param_pointer) = atoi(key);}
            else if (param_array[i].param_type == PARAM_TYPE_FLOAT)
                {*((float*) param_array[i].param_pointer) = strtof(key, NULL);}
            else if (param_array[i].param_type == PARAM_TYPE_DOUBLE)
                {*((double*) param_array[i].param_pointer) = strtod(key, NULL);}
            else if (param_array[i].param_type == PARAM_TYPE_CHAR)
                {strcpy((char*) param_array[i].param_pointer, (const char*) key);}
            else
            {
                printf("ERROR: Parameter %s does not have a valid parameter type specified!\n", param_array[i].param_name);
                return_value = CONFIGURATION_FROM_FILE_FAILED;
            }
        }
    }

    // xmlFreeDoc(infile);
    return return_value;
}


void read_from_input_file(const char *filename, param_array_t param_array[], int length_array, int *status)
{
    if(filename == NULL)
    {
        *status = CONFIGURATION_FROM_FILE_FAILED;
        printf("ERROR: file name is empty!\n");
    }    
    else
    {
        const char* extension = strrchr(filename, '.');
        
        if(strcmp(extension, ".json") == 0)
        {
            *status = read_from_json_file(filename, param_array, length_array);
        }
        else if(strcmp(extension, ".xml") == 0)
        {
            *status = read_from_XML_file(filename, param_array, length_array);
        }
        else
        {
            printf("ERROR: Input file '%s' has an invalid extension\n", filename);
            *status = CONFIGURATION_FROM_FILE_FAILED;
        }
    }

    if(*status == CONFIGURATION_FROM_FILE_FAILED)
        {printf("ERROR: Input file '%s' was not read succesfully\n", filename);}

}
