
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <string>

using namespace std; //Para no tener que poner std:: antes de: vectores, strings, cout, cin, endl, isnan, etc...


#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cmath>


__constant__ float d_umbral_2; //EL umbral en memoria constante que pide el ejercicio 2

#define MAX_LINE 2048 //Valor tan grande para intentar asegurar de que quepa la fila/linea entera

#define MAX_TAIL_NUM 8 //Para poder operar con la matriculas, GPU no admite strings, neceistaremos este maximo para poder tener 
                       //arrays/vectores de caracteres (es 8 porque la longitud de matricula arece variar y el maximo parece ser 6,
                       // asi que dejamos algo de posible margen.

//COLUMNAS DEL DATASET QUE VAMOS A LEER (Añadir mas segun se necesite)
#define COL_ARR_DELAY 12
#define COL_DEP_DELAY 10
#define COL_TAIL_NUM 3



//SI HACE FALTA AÑADIR MAS FUNCIONES COMO ESTAS
// Convierte a float o devuelve NAN si no hay valor
float parseFloat(const char* token) {
    if (token == NULL || strlen(token) == 0 || token[0] == '\n') { //Todas las posibilidades para valor nulo
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


//LECTURA CSV (Se deben pasar la dirección de los vectores donde copiar las columnas)
void leerCSV(const string& ruta,
    vector<float>& arrDelay,
    vector<float>& depDelay,
    vector<string>& tailNum,
    int maxFilas) {

    FILE* file = fopen(ruta.c_str(), "r"); //Abre el archivo

    if (file == NULL) {
        printf("Error al abrir el archivo\n");
        exit(1);
    }

    char line[MAX_LINE]; //para pillar las filas

    //Salta la cabecera
    fgets(line, sizeof(line), file);

    int filasLeidas = 0;

    while (fgets(line, sizeof(line), file)) { //Leer cada linea

        if (maxFilas > 0 && filasLeidas >= maxFilas) //Para poder cargar pocas filas, poner maxFilas a 0 para cargarlo todo
            break;

        char* token = strtok(line, ","); //Separa por las comas
        int column = 0;

        //Inicializamos a NAN o vacio por si no se leen o encuentran en el fichero
        float arr = NAN;
        float dep = NAN;
        string tail = "";

        while (token != NULL) { //Recorre cada columna (hasta el final, que sera NULL)

            //ESTOS IFS DE ABAJO ASEGURAN QUE SOLO GUARDAMOS Y LEEMOS LO NECESARIO (para que vamos a cargar cosas que no queremos)
            //Si queremos guardar mas columnas -> añadir mas ifs
            if (column == COL_ARR_DELAY) {
                arr = parseFloat(token);
            }

            if (column == COL_DEP_DELAY) {
                dep = parseFloat(token);
            }

            if (column == COL_TAIL_NUM) {
                if (token != NULL)
                    tail = token;
            }

            //Pasamos a la siguiente columna
            token = strtok(NULL, ","); 
            column++;
        }

        //Añado los valores de la columna al final de su vector correspondiente
        arrDelay.push_back(arr);
        depDelay.push_back(dep);
        tailNum.push_back(tail);

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

    //SI ESO AÑADIR ALGO PARA QUE IMPRIMA SI NO SE HAN ENCONTRADO VALORES DENTRO DEL UMBRAL
}


//Funcion para el apartado 1: Calculo de adelantos u atrasos en la llegada y almacenamiento de matriculas y tiempos:
__global__ void detectarAterrizajes(float* arrDelay, char* tailNum, int N, float* outDelay, char* outTail, int* contador) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N && !isnan(arrDelay[i])) {

        bool cumple = false;

        if (d_umbral_2 >= 0) {//retraso
            if (arrDelay[i] >= d_umbral_2)
                cumple = true;
        }
        else {//adelanto
            if (arrDelay[i] <= d_umbral_2)
                cumple = true;
        }

        if (cumple) {

            int pos = atomicAdd(contador, 1); //Como nos devuelve el valor anterior no esta devolviendo la posicion adecuada, empezando por el 0

            outDelay[pos] = arrDelay[i]; //Copiamos el valor nuemrico del retraso en el vector que los guarda

            for (int j = 0; j < MAX_TAIL_NUM; j++) {
                outTail[pos * MAX_TAIL_NUM + j] = tailNum[i * MAX_TAIL_NUM + j]; //Hacemos los mismo con la matricula, en este caso copiamos
                                                                                 //caracter a caracter cada matricula
            }

            if (d_umbral_2 >= 0) {//retraso
                printf("- Hilo: #%d | Matricula: %s | Retraso: %.2f min\n", i, &tailNum[i * MAX_TAIL_NUM], arrDelay[i]);
            }
            else {//adelanto
                printf("- Hilo: #%d | Matricula: %s | Adelanto: %.2f min\n", i, &tailNum[i * MAX_TAIL_NUM], abs(arrDelay[i]));
            }
            
        }
    }
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
    vector<string> tailNum;

    int limite = 0;  //Cambiar para cargar mas o menos datos (0 para cargarlos todos)

    if (ruta == "") {
        cout << "\nCargando con ruta por defecto\n";
        //FUNCION DE CARGA CON LA RUTA POR DEFECTO
        leerCSV("D:/Fichero PAP/Airline_dataset.csv", arrDelay, depDelay, tailNum, limite);
    }
    else
    {
        cout << "\nCargando con ruta: " << ruta << "\n" << endl;
        //FUNCION DE CARGA CON LA RUTA ESPECIFICADA
        leerCSV(ruta, arrDelay, depDelay, tailNum, limite);
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

            float umbral;
            cout << "Introduzca el umbral (positivo para retrasos, negativo para adelantos): ";
            cin >> umbral;

            int N = arrDelay.size();


            //Convertir los strings a array plano (para que la GPU los pueda usar)
            char* tailNumPlano = new char[N * MAX_TAIL_NUM]; 
            //Reserva un bloque de memoria continuo capaz de guardar N matrículas, cada una de MAX_TAIL_NUM caracteres

            for (int i = 0; i < N; i++) {
                strncpy(&tailNumPlano[i * MAX_TAIL_NUM], tailNum[i].c_str(), MAX_TAIL_NUM); //Copia lo que tailNum tiene en la posicion i,
                                                                                            // en forma de caracteres con longitud de
                                                                                            // MAX_TAIL_NUM. Ej: hol -> h, o, l, ?, ?, ... 
                                                                                            // hasta MAX_TAIL_NUM
                tailNumPlano[i * MAX_TAIL_NUM + MAX_TAIL_NUM - 1] = '\0'; //Añadimos esto para que cuando imprimimos se pare en cada
                                                                          // matricula correspondiente en vez de imprimir todo el array que 
                                                                          // contiene a los caracteres de las matriculas
            }


            //Punteros para GPU
            float* d_arrDelay;
            char* d_tailNum;
            float* d_outDelay;
            char* d_outTail;
            int* d_contador;

            //Reserva de memoria
            cudaMalloc(&d_arrDelay, N * sizeof(float));
            cudaMalloc(&d_tailNum, N * MAX_TAIL_NUM);
            cudaMalloc(&d_outDelay, N * sizeof(float));
            cudaMalloc(&d_outTail, N * MAX_TAIL_NUM);
            cudaMalloc(&d_contador, sizeof(int));

            cudaMemset(d_contador, 0, sizeof(int)); //Inicializamos el contador a 0

            //Copiado a memoria
            cudaMemcpy(d_arrDelay, arrDelay.data(), N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tailNum, tailNumPlano, N * MAX_TAIL_NUM, cudaMemcpyHostToDevice);

            //Memoria constante
            cudaMemcpyToSymbol(d_umbral_2, &umbral, sizeof(float));


            //Configuracion de bloques e hilos
            dim3 blocksInGrid;
            dim3 threadsInBlock;

            configurarKernel(N, blocksInGrid, threadsInBlock); //Llamamos a la funcion de configuracion

            cout << "\nProcediendo a la ejecucion, espere por favor...\n";

            detectarAterrizajes <<<blocksInGrid, threadsInBlock>>> (d_arrDelay, d_tailNum, N, d_outDelay, d_outTail, d_contador);

            cudaDeviceSynchronize(); //Esperamos a que todos los hilos terminen

            //Recuperar los resultados
            int h_contador;
            cudaMemcpy(&h_contador, d_contador, sizeof(int), cudaMemcpyDeviceToHost);

            vector<float> outDelay(h_contador);
            vector<char> outTail(h_contador * MAX_TAIL_NUM);

            cudaMemcpy(outDelay.data(), d_outDelay, h_contador * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(outTail.data(), d_outTail, h_contador * MAX_TAIL_NUM, cudaMemcpyDeviceToHost);

            cout << "\nSe han encontrado: " << h_contador << " aviones" << endl;

            //Imprimir resulatdos con los vectores
            int cont = 0;
            if (umbral > 0) {
                while (cont < h_contador) {
                    cout << "- Matricula " << &outTail[cont * MAX_TAIL_NUM] << " Retraso: " << outDelay[cont] << " minutos" << endl;
                    cont++;
                }
            }
            else {
                while (cont < h_contador) {
                    cout << "Matricula " << &outTail[cont * MAX_TAIL_NUM] << " Adelanto: " << abs(outDelay[cont]) << " minutos" << endl;
                    cont++;
                }
            }
            

            //Liberar memoria
            cudaFree(d_arrDelay);
            cudaFree(d_tailNum);
            cudaFree(d_outDelay);
            cudaFree(d_outTail);
            cudaFree(d_contador);

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
