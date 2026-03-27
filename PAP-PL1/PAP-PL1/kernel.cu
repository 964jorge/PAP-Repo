
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>

using namespace std; //Para no tener que poner std:: antes de: vectores, strings, cout, cin, endl, isnan, etc...


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
void leerCSV(const string& ruta,
    vector<float>& arrDelay,
    vector<float>& depDelay,
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

    cout << "\nFilas cargadas: " << filasLeidas << endl;
}


//Funcion para la configuracion de los bloques e hilos que se pide aplicar en cada apartado
void configurarKernel(int N, dim3& blocks, dim3& threads) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxThreads = prop.maxThreadsPerBlock;

    //int threadsX = (maxThreads < 256) ? maxThreads : 256; //Si queremos limitarlo a 256 descomentar y comentar lo de abajo
    int threadsX = maxThreads;

    //Configuracion
    threads = dim3(threadsX);
    blocks = dim3((N + threadsX - 1) / threadsX);

    //cout << "Threads por bloque: " << threads.x << endl; (En orednador de Jose 1024)
    //cout << "Numero de bloques: " << blocks.x << endl;
}



//Funcion para el apartado 1: Calculo de adelantos u atrasos en la salida:
__global__ void detectarRetrasos(float* depDelay, int longitud, float umbral) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < longitud && !isnan(depDelay[i])) {

        if (umbral >= 0) {
            //retraso
            if (depDelay[i] >= umbral) {
                printf("- Hilo #%d: Retraso de %.2f minutos\n", i, depDelay[i]);
            }
        }
        else {
            //adelanto
            if (depDelay[i] <= umbral) {
                printf("- Hilo #%d: Adelanto de %.2f minutos\n", i, abs(depDelay[i]));
            }
        }
    }

    //SI ESO AÑADIR ALGO PARA QUE IMPRIMA SI NO SE HAN ENCONTRADO VALORES DENTRO DEL UMBRAL (en el 2 tiene pinta de ser mas facil)
}






int main()
{

    string ruta = "";

    cout << "\nEL1 PAP 2026 Jorge Lopez y Jose Antonio Lopez\n";
    cout << "Introduzca la ruta base del dataset (o pulse intro para usar C:/Airline_dataset.csv): ";
    getline(cin, ruta);


    //Vectores que vamos a necesitar
    vector<float> arrDelay;
    vector<float> depDelay;

    int limite = 0;  //Cambiar para cargar mas o menos datos (0 para cargarlos todos)

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
        case 1: {

            float umbral;
            cout << "Introduzca el umbral (positivo para retrasos, negativo para adelantos): ";
            cin >> umbral;

            // Copiar a GPU
            float* d_depDelay;
            int N = depDelay.size(); //devuelve el tamaño del vector

            cudaMalloc(&d_depDelay, N * sizeof(float)); //Reservamos la memoria
            //depDelay.data() nos devuelve el puntero al primer elemento del vector
            cudaMemcpy(d_depDelay, depDelay.data(), N * sizeof(float), cudaMemcpyHostToDevice);


            //Configuracion de bloques e hilos
            dim3 blocksInGrid;
            dim3 threadsInBlock;

            configurarKernel(N, blocksInGrid, threadsInBlock); //Llamamos a la funcion de configuracion

            cout << "\nProcediendo a la ejecucion, espere por favor...\n";


            //Lanzamos todos los hilos a ejecutar el programa
            detectarRetrasos <<<blocksInGrid, threadsInBlock>>> (d_depDelay, N, umbral); 

            cudaDeviceSynchronize(); //Esperamos a que todos los hilos terminen

            cudaFree(d_depDelay); //Liberamos la memoria

            break;
        }

        case 2: {
            cout << "\nProcediendo a la ejecucion 2, espere por favor...\n";
            break;
        }

        case 3: {
            cout << "\nProcediendo a la ejecucion 3, espere por favor...\n";
            break;
        }

        case 4: {
            cout << "\nProcediendo a la ejecucion 4, espere por favor...\n";


            //ESTA FUNCION ES SOLO PARA COMPROBAR, ELIMINAR CUANDO SE QUIERA HACER EL APARTADO 4:
            for (int i = 0; i < 10 && i < arrDelay.size(); i++) {

                cout << "Fila " << i << " | ";

                if (isnan(arrDelay[i]))
                    cout << "ArrDelay: NAN | ";
                else
                    cout << "ArrDelay: " << arrDelay[i] << " | ";

                if (isnan(depDelay[i]))
                    cout << "DepDelay: NAN | ";
                else
                    cout << "DepDelay: " << depDelay[i] << " | ";

                cout << endl;
            }


            break;
        }

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
