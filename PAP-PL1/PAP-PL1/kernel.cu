
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;


#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cmath>

#define MAX_LINE 2048 //Valor tan grande para intentar asegurar de que quepa la fila/linea entera


//COLUMNAS DEL DATASET QUE VAMOS A LEER (Añadir mas segun se necesite)
#define COL_ARR_DELAY 12
#define COL_DEP_DELAY 10


//SI HACE FALTA AÑADIR MAS FUNCIONES COMO ESTAS
// Convierte a float o devuelve NAN si no hay valor
float parseFloat(const char* token) {
    if (token == NULL || strlen(token) == 0 || token[0] == '\n') {
        return NAN;
    }
    return atof(token);
}

//Convierte a int o devuelve NAN (como float)
float parseIntAsFloat(const char* token) {
    if (token == NULL || strlen(token) == 0 || token[0] == '\n') {
        return NAN;
    }
    return (float)atoi(token);
}


//LECTURA CSV (Se deben pasar la direccion de los vectores donde copiar las columnas)
void leerCSV(const std::string& ruta,
    std::vector<float>& arrDelay,
    std::vector<float>& depDelay,
    int maxFilas) {

    FILE* file = fopen(ruta.c_str(), "r"); //Abre el archivo

    if (file == NULL) {
        printf("Error al abrir el archivo\n");
        exit(1);
    }

    char line[MAX_LINE];

    //Salta la cabecera
    fgets(line, sizeof(line), file);

    int filasLeidas = 0;

    while (fgets(line, sizeof(line), file)) { //Leer cada linea

        if (maxFilas > 0 && filasLeidas >= maxFilas) //Para poder cargar pocas filas, poner maxFilas a 0 para cargarlo todo
            break;

        char* token = strtok(line, ","); //Separa por las comas
        int column = 0;

        //Inicializamos a NAN por si no se leem o encuentran el el fichero
        float arr = NAN;
        float dep = NAN;

        while (token != NULL) { //Recorre cada columna (hasta el final, que sera NULL)

            //ESTOS IFS DE ABAJO ASEGURAN QUE SOLO GUARDAMOS Y LEEMOS LO NECESARIO (para que vamos a cargar cosas que no queremos)
            //Si queremos guardar mas columnas -> añadir mas ifs
            if (column == COL_ARR_DELAY) {
                arr = parseFloat(token);
            }

            if (column == COL_DEP_DELAY) {
                dep = parseFloat(token);
            }

            token = strtok(NULL, ","); //Pasamos a la siguiente columna
            column++;
        }

        //Añado los valores de la columna al final de su vector correspondiente
        arrDelay.push_back(arr);
        depDelay.push_back(dep);

        filasLeidas++;
    }

    fclose(file); //Cierro el fichero

    std::cout << "\nFilas cargadas: " << filasLeidas << std::endl;
}





int main()
{

    string ruta = "";

    cout << "\nEL1 PAP 2026 Jorge Lopez y Jose Antonio Lopez\n";
    cout << "Introduzca la ruta base del dataset (o pulse intro para usar C:/Airline_dataset.csv): ";
    getline(cin, ruta);


    //Vectores que vamos a necesitar
    std::vector<float> arrDelay;
    std::vector<float> depDelay;

    int limite = 0;  //Cambiar para acargar mas o menos datos

    if (ruta == "") {
        cout << "\nCargando con ruta por defecto\n";
        //FUNCION DE CARGA CON LA RUTA POR DEFECTO
        leerCSV("D:/Fichero PAP/Airline_dataset.csv", arrDelay, depDelay, limite);
    }
    else
    {
        cout << "\nCargando con ruta: " << ruta << "\n" << endl;
        //FUNCION DE CARGA CON LA RUTA ESPECIFICADA
        leerCSV(ruta, arrDelay, depDelay, limite);
    }


    int opcion;
    bool ejecutar = true;

    while (ejecutar) {
        cout << "\n--- MENU DE OPCIONES ---\n";
        cout << "1. Retraso en despegues\n";
        cout << "2. Retraso en aterrizajes\n";
        cout << "3. Reduccion de retraso\n";
        cout << "4. Histograma de aeropuertos\n";
        cout << "5. Salir\n";
        cout << "Elija la opcion: ";
        cin >> opcion;

        //Switch case sencillo para manejar las 5 opciones posibles
        switch (opcion) {
        case 1:
            cout << "\nProcediendo a la ejecucion 1, espere por favor...\n";


            for (int i = 0; i < 10 && i < arrDelay.size(); i++) {

                std::cout << "Fila " << i << " | ";

                if (std::isnan(arrDelay[i]))
                    std::cout << "ArrDelay: NAN | ";
                else
                    std::cout << "ArrDelay: " << arrDelay[i] << " | ";

                if (std::isnan(depDelay[i]))
                    std::cout << "DepDelay: NAN | ";
                else
                    std::cout << "DepDelay: " << depDelay[i] << " | ";

                std::cout << std::endl;
            }


            break;

        case 2:
            cout << "\nProcediendo a la ejecucion 2, espere por favor...\n";
            break;

        case 3:
            cout << "\nProcediendo a la ejecucion 3, espere por favor...\n";
            break;

        case 4:
            cout << "\nProcediendo a la ejecucion 4, espere por favor...\n";
            break;

        case 5:
            cout << "\nSaliendo...\n";
            ejecutar = false;   //Terminamos el bucle
            break;

        //SI SE QUIERE DAR LA OPCION DE CAMBIAR LA RUTA DE ACCESO BASTARIA CON AÑADIR OTRO CASE MAS

        default:
            cout << "\nOpcion no valida\n";
            //Para el caso de que no pusiera un numero
            cin.clear(); //Limpia errores
            cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
            break;
        }
    }




    return 0;
}
