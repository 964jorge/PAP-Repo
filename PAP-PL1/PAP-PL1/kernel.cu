
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>


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
#define COL_WEATHER_DELAY 13
#define COL_ARR_TIME 11
#define COL_DEP_TIME 9
#define COL_TAIL_NUM 3
#define COL_ORIGIN_AIRPORT 6
#define COL_DEST_AIRPORT 8
#define COL_ORIGIN_SEQ_ID 5
#define COL_DEST_SEQ_ID 7



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

float parseInt(const char* token) {
    if (token == NULL || strlen(token) == 0 || token[0] == '\n') { //Todas las posibilidades para valor nulo
        return NAN;
    }
    return atoi(token);
}


//LECTURA CSV (Se deben pasar la dirección de los vectores donde copiar las columnas)
void leerCSV(const string& ruta,
    vector<float>& arrDelay,
    vector<float>& depDelay,
    vector<float>& weatherDelay,
    vector<float>& arrTime,
    vector<float>& depTime,
    vector<string>& tailNum,
    vector<string>& originAirport,
    vector<string>& destAirport,
    vector<int>& originID,
    vector<int>& destID,
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
        float weath = NAN;
        float arrTi = NAN;
        float depTi = NAN;
        string tail = "";
        string orAir = "";
        string destAir = "";
        int orID = NAN;
        int deID = NAN;

        while (token != NULL) { //Recorre cada columna (hasta el final, que sera NULL)

            //ESTOS IFS DE ABAJO ASEGURAN QUE SOLO GUARDAMOS Y LEEMOS LO NECESARIO (para que vamos a cargar cosas que no queremos)
            //Si queremos guardar mas columnas -> añadir mas ifs
            if (column == COL_ARR_DELAY) {
                arr = parseFloat(token);
            }


            if (column == COL_DEP_DELAY) {
                dep = parseFloat(token);
            }

            if (column == COL_WEATHER_DELAY) {
                weath = parseFloat(token);
            }

            if (column == COL_ARR_TIME) {
                arrTi = parseFloat(token);
            }

            if (column == COL_DEP_TIME) {
                depTi = parseFloat(token);
            }

            if (column == COL_TAIL_NUM) {
                if (token != NULL)
                    tail = token;
            }

            if (column == COL_ORIGIN_AIRPORT) {
                if (token != NULL)
                    orAir = token;
            }

            if (column == COL_DEST_AIRPORT) {
                if (token != NULL)
                    destAir = token;
            }

            if (column == COL_ORIGIN_SEQ_ID) {
            
                orID = parseInt(token);

            
            }

            if (column == COL_DEST_SEQ_ID) {

                deID = parseInt(token);


            }

            //Pasamos a la siguiente columna
            token = strtok(NULL, ","); 
            column++;
        }

        //Añado los valores de la columna al final de su vector correspondiente y hace 0 los NAN


        arrDelay.push_back(arr);
        depDelay.push_back(dep);
        weatherDelay.push_back(weath);
        arrTime.push_back(arrTi);
        depTime.push_back(depTi);
        tailNum.push_back(tail);
        originAirport.push_back(orAir);
        destAirport.push_back(destAir);
        originID.push_back(orID);
        destID.push_back(deID);

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







__global__ void reductorMaximalSimple(int* datos, int* resultado, int tamanno) {

    //aqui cada hilo que tiene datos (por eso lo de < que tamanno) hace el maximo atomicamente entre el resultado y su posicion

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < tamanno) {
    
        atomicMax(resultado, datos[idx]);

    }

}

__global__ void reductorMaximalBasico(int* datos, int* resultado, int tamanno) {

    //empezamos declarando la memoria compartida de cada bloque

    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //si el hilo tiene datos los copia en la posicion del bloque que le corresponde
    if (idx < tamanno) {
    
        datosEnBloque[threadIdx.x] = datos[idx];


    }

    //esperamos a todos    
    __syncthreads();

    //definimos un maximo local, que sera el mayor de cada bloque
    int maximoLocal;

    //ahora, si eres el primer hilo, mirate a ti y al siguiente
    //si eres el ultimo mirate a ti y al anterior
    //si no eres ninguno mirate a ti, a tu anterior y a tu posterior
    if (threadIdx.x == 0){
    

        maximoLocal = max(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]);
    
    }else if(threadIdx.x == blockDim.x - 1) {
     

        maximoLocal = max(datosEnBloque[threadIdx.x - 1], datosEnBloque[threadIdx.x]);
    
    }else {
    
        maximoLocal = max(datosEnBloque[threadIdx.x - 1], max(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]));
    
    }
    
    

    //ahora comparad el maximo local con el resultado y dejad el mayor

    atomicMax(resultado, maximoLocal);


}



__global__ void reductorMaximalIntermedio(int* datos, int* resultado, int tamanno) {


    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < tamanno) {

        datosEnBloque[threadIdx.x] = datos[idx];


    }

    __syncthreads();


    if (threadIdx.x == 0) {


        datosEnBloque[threadIdx.x] = max(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]);

    }
    else if (threadIdx.x == blockDim.x - 1) {


        datosEnBloque[threadIdx.x] = max(datosEnBloque[threadIdx.x - 1], datosEnBloque[threadIdx.x]);

    }
    else {

        datosEnBloque[threadIdx.x] = max(datosEnBloque[threadIdx.x - 1], max(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]));

    }


    __syncthreads();

    //hasta aqui es identico al maximal basico

    //que hacemos aqui, si eres un hilo par que no sea el ultimo mirate a ti y al siguiente y comparalo con resultado
    //si eres el ultimo mirate a ti y comparate con el resultado

    if (idx % 2 == 0) {
    

        if (threadIdx.x == blockDim.x - 1 || idx == tamanno - 1) {
        
            atomicMax(resultado, datosEnBloque[threadIdx.x]);
        
        }
        else {
        
            atomicMax(resultado, max(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]));
        
        }

    
    }




}




__global__ void reductorMaximalReductor(int* datos, int* resultado, int tamanno) {

    //declaramos la memoria compartida, e inicializamos todos los valores con el valor que tenga el hilo y si no lo tiene con el minimo de los enteros. luego esperamos a todos

    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    datosEnBloque[threadIdx.x] = (idx < tamanno) ? datos[idx] : INT_MIN;

    __syncthreads();

    

    //eres el hilo 0 y el tamaño de bloques es impar? comparate con el ultimo elemento y guardalo en tu posicion

    if(threadIdx.x == 0 && blockDim.x % 2 != 0){ 
        
        datosEnBloque[0] = max(datosEnBloque[0], datosEnBloque[blockDim.x - 1]);
    
    }

    //Importante, en cada iteracion esperamos a todos los hilos
    //cada hilo de la primera mitad se va a comparar con el que tenga su indice + el stride, esto hace que la solucion se encuentre tras cada paso en los elementos menores al stride
    //y como cada vez el stride se corta a la mitad va a tender a la primera posicion

    for (int stride = blockDim.x / 2; stride > 0; stride = stride/2) {
    
        
        if (threadIdx.x < stride) {

            datosEnBloque[threadIdx.x] = max(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + stride]);

        }


        __syncthreads();

    
    }

     //luego despues de tener el resultado en la posicion 0 si eres el hilo 0 copia el resultado en el indice igual al bloque (porque solo hay un hilo 0 por bloque)
    if (threadIdx.x == 0) {

        resultado[blockIdx.x] = datosEnBloque[0];

    }


}











__global__ void reductorMinimalSimple(int* datos, int* resultado, int tamanno) {

    //Es igual que el maximal simple pero con el minimo atomico en vez del maximo atomico

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < tamanno) {
    
        atomicMin(resultado, datos[idx]);

    }
    

}




__global__ void reductorMinimalBasico(int* datos, int* resultado, int tamanno) {


    //Es practicamente identico al maximal basico

    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < tamanno) {

        datosEnBloque[threadIdx.x] = datos[idx];

    }

   
    __syncthreads();


    int minimoLocal;


    if (threadIdx.x == 0) {


        minimoLocal = min(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]);

    }
    else if (threadIdx.x == blockDim.x - 1) {


        minimoLocal = min(datosEnBloque[threadIdx.x - 1], datosEnBloque[threadIdx.x]);

    }
    else {

        minimoLocal = min(datosEnBloque[threadIdx.x - 1], min(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]));

    }





    atomicMin(resultado, minimoLocal);

}



__global__ void reductorMinimalIntermedio(int* datos, int* resultado, int tamanno) {

    //practicamente identico al maximal intermedio

    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < tamanno) {

        datosEnBloque[threadIdx.x] = datos[idx];

    }

    __syncthreads();


    if (threadIdx.x == 0) {


        datosEnBloque[threadIdx.x] = min(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]);

    }
    else if (threadIdx.x == blockDim.x - 1) {


        datosEnBloque[threadIdx.x] = min(datosEnBloque[threadIdx.x - 1], datosEnBloque[threadIdx.x]);

    }
    else {

        datosEnBloque[threadIdx.x] = min(datosEnBloque[threadIdx.x - 1], min(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]));

    }


    __syncthreads();



    if (idx % 2 == 0) {


        if (threadIdx.x == blockDim.x - 1 || idx == tamanno - 1) {

            atomicMin(resultado, datosEnBloque[threadIdx.x]);

        }
        else {

            atomicMin(resultado, min(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + 1]));

        }


    }




}



__global__ void reductorMinimalReductor(int* datos, int* resultado, int tamanno) {


    //practicamente identico al maximal reductor

    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    datosEnBloque[threadIdx.x] = (idx < tamanno) ? datos[idx] : INT_MAX;

    __syncthreads();


    if (threadIdx.x == 0 && blockDim.x % 2 != 0) {

        datosEnBloque[0] = min(datosEnBloque[0], datosEnBloque[blockDim.x - 1]);

    }


    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {


        if (threadIdx.x < stride) {

            datosEnBloque[threadIdx.x] = min(datosEnBloque[threadIdx.x], datosEnBloque[threadIdx.x + stride]);

        }


        __syncthreads();


    }


    if (threadIdx.x == 0) {

        resultado[blockIdx.x] = datosEnBloque[0];

    }

}











//Este es el lanzador el ejercicio 3.  Le estamos pasando la opcion que nos indica sobre que conjunto de datos trabajar y la opcion de que operacion hacer (ademas de todos los datos posibles)
void lanzadorReductor(int opcion1, int opcion2, vector<float>& depDelay, vector<float>& arrDelay, vector<float>& weatherDelay, vector<float>& depTime, vector<float>& arrTime) {

    //inicializamos cosas como los punteros para el kernel
    vector<int> vectorDatos;
    int* d_vectorDatos;
    int* resultado;
    int* resultadoReductor;
    char* columna;

    //este switch lo hacemos para ver que ha pedido el usuario, en columna guardamos que columna guardamos luego para el print y 
    //tomamos de los valores que pasamos los que son numeros truncados como se pide llenando nuestro vectorDatos que sera el que usaremos para inicializar el puntero del kernel
    switch (opcion1) {

    case 1: {

        columna = "DEP_DELAY";
   
        for (int i = 0; i < size(depDelay); i++) {
        
            if (!isnan(depDelay[i])) {

                vectorDatos.push_back((int)truncf(depDelay[i]));

            }
            

        }

        break;

    }

    case 2: {
   
        columna = "ARR_DELAY";

        for (int i = 0; i < size(arrDelay); i++) {

            if (!isnan(arrDelay[i])) {

                vectorDatos.push_back((int)truncf(arrDelay[i]));

            }


        }

        break;
    
    }

    case 3: {
    
        columna = "WEATHER_DELAY";

        for (int i = 0; i < size(weatherDelay); i++) {

            if (!isnan(weatherDelay[i])) {

                vectorDatos.push_back((int)truncf(weatherDelay[i]));

            }
            

        }

        break;
    
    }

    case 4: {

        columna = "DEP_TIME";

        for (int i = 0; i < size(depTime); i++) {

            if (!isnan(depTime[i])) {

                vectorDatos.push_back((int)truncf(depTime[i]));

            }


        }

        break;
    
    }

    case 5: {

        columna = "ARR_TIME";

        for (int i = 0; i < size(arrTime); i++) {

            if (!isnan(arrTime[i])) {

                vectorDatos.push_back((int)truncf(arrTime[i]));

            }


        }

        break;
    
    }


    }
    

    //ahora tenemos vectorDatos con los datos ya casteados a entero truncado entonces el tamaño del problema sera el del vector
    //inicializamos varianbles para luego poder hacer prints y tabajar con ellas
    int resultadoAImprimir;
    int tamanno = vectorDatos.size();
    vector<int> resultadoReduccionVector;

    //definimos el tamaño de bloque y el numero de ellos con nuestra funcion
    dim3 blocksInGrid;
    dim3 threadsInBlock;
    size_t espacio = vectorDatos.size() * sizeof(int);
    configurarKernel(espacio, blocksInGrid, threadsInBlock);
    
    //alocamos para luego poder traernos a CPU el resultado de la reduccion (que va a ser unico en la MAYORIA de casos)
    cudaMalloc(&resultado, sizeof(int));

    //alocamos y copiamos en el puntero el vector de datos que ya teniamos 
    cudaMalloc(&d_vectorDatos, espacio);
    cudaMemcpy(d_vectorDatos, vectorDatos.data(), espacio, cudaMemcpyHostToDevice);

    //este sera para el resulado del ultimo tipo de reductor, el que trae un vector de datos combinados por bloque, no se usara hasta el ultimo caso pero por tenerlo alocado ya
    //como vamos a combinar todos los bloques en uno necesitamos que el resultado tenga el tamaño de un bloque
    cudaMalloc(&resultadoReductor, blocksInGrid.x * sizeof(int));
    

    //este es el selector de antes que nos decia que operacion hacer, la primera es maximizar, la segunda minimizar
    if (opcion2 == 1) {
    
       
        //este sera para inicializar el resultado que aun no tenemos y devolvera el kernel. necesitamos que apunte a algo para poder hacer operaciones atomicas sobre ello y como estamos maximizando 
        //cualquier valor sera mayor o igual que el menor de los enteros entonces no interfiere con los datos que pasamos
        int valorInicial = INT_MIN;
        


        //ESTOS SON LOS PRIMEROS 3 SUBAPARTADOS


        //Estos son bastante iguales, lo que cambia es el kernel que llamamos pero todos 'hacen lo mismo'
        //inicializamos el puntero resultado al valor inicial, lanzamos el kernel, esperamos a que acabe, traemos el resultado a CPU e imprimimos

        cudaMemcpy(resultado, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
        reductorMaximalSimple <<<blocksInGrid, threadsInBlock >>> (d_vectorDatos, resultado, tamanno);
        cudaDeviceSynchronize();
        cudaMemcpy(&resultadoAImprimir, resultado, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n[Maximizacion Simple] %s %d\n", columna, resultadoAImprimir);

        //Aqui introducimos la memoria compartida, le pasamos de tamaño el tamaño del bloque (a ambos subapartados)

        cudaMemcpy(resultado, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
        reductorMaximalBasico<<<blocksInGrid, threadsInBlock, threadsInBlock.x*sizeof(int)>>> (d_vectorDatos, resultado, tamanno);
        cudaDeviceSynchronize();
        cudaMemcpy(&resultadoAImprimir, resultado, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n[Maximizacion Basica] %s %d\n", columna, resultadoAImprimir);


        cudaMemcpy(resultado, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
        reductorMaximalIntermedio <<<blocksInGrid, threadsInBlock, threadsInBlock.x * sizeof(int) >>> (d_vectorDatos, resultado, tamanno);
        cudaDeviceSynchronize();
        cudaMemcpy(&resultadoAImprimir, resultado, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n[Maximizacion Intermedia] %s %d\n", columna, resultadoAImprimir);




        //ESTE ES EL ULTIMO SUBAPARTADO



        //aqui es donde se complica la cosa, vamos a ir poco a poco
        //inicializamos el puntero al valor inicial de antes, este sera el del resultado por el momento, para las siguientes iteraciones se usaran los que salgan de resultado del kernel
        int* arrInicial = new int[blocksInGrid.x];
        for (int i = 0; i < blocksInGrid.x; i++) { arrInicial[i] = valorInicial; }


        //pasamos el array al resultado y lanzamos el kernel
        cudaMemcpy(resultadoReductor, arrInicial, blocksInGrid.x * sizeof(int), cudaMemcpyHostToDevice);
        reductorMaximalReductor <<<blocksInGrid, threadsInBlock, threadsInBlock.x * sizeof(int) >>> (d_vectorDatos, resultadoReductor, tamanno);
        cudaDeviceSynchronize();
        //importante el resize porque si no puede ser que no haya espacio suficiente (aqui un poco mas complicado que pase pero mas adelante crucial)
        resultadoReduccionVector.resize(blocksInGrid.x);
        //nos traemos a cpu el resultado y liberamos el puntero porque no lo necesitamos mas hasta luego
        cudaMemcpy(resultadoReduccionVector.data(), resultadoReductor, blocksInGrid.x * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(resultadoReductor);
        
        //aqui esta el bucle que nos dice que si el resultado es de mas de 10 elementos seguimos lanzando kernels para que la GPU procese casi todo
        while (resultadoReduccionVector.size() > 10) {
        
            //definimos el tamaño del problema otra vez, ahora sera el tamaño del resultado que es lo que vamos a pasar al puntero del kernel
            dim3 blocksInGrid;
            dim3 threadsInBlock;
            size_t espacio = resultadoReduccionVector.size() * sizeof(int);
            int tamanno = resultadoReduccionVector.size(); 

            configurarKernel(espacio, blocksInGrid, threadsInBlock);

            //alocamos y copiamos el nuevo resultado (no es lo del kernel anterior, es el resultado que hay que pasarle al kernel para que lo guarde ahi)
            cudaMalloc(&resultadoReductor, blocksInGrid.x * sizeof(int));
            cudaMemcpy(resultadoReductor, &valorInicial, blocksInGrid.x * sizeof(int), cudaMemcpyHostToDevice);

            //aqui el puntero nuevos datos es el que guardara el resultado de la iteracion anterior para poder pasarlo a kernel
            int* d_nuevosDatos;
            cudaMalloc(&d_nuevosDatos, espacio);
            cudaMemcpy(d_nuevosDatos, resultadoReduccionVector.data(), espacio, cudaMemcpyHostToDevice);

            //lamzamos kernel y esperamos a que acabe (tamanno no es el de antes, es el nuevo que hemos definido con el tamaño de los resultados)
            reductorMaximalReductor <<< blocksInGrid, threadsInBlock, threadsInBlock.x * sizeof(int) >>> (d_nuevosDatos, resultadoReductor, tamanno);
            cudaDeviceSynchronize();

            //aqui es donde es critico el resize porque si no puede pasar lo de que no haya suficiente espacio. copiamos el resultado en resultadoReduccionVector
            resultadoReduccionVector.resize(blocksInGrid.x);
            cudaMemcpy(resultadoReduccionVector.data(), resultadoReductor, blocksInGrid.x * sizeof(int), cudaMemcpyDeviceToHost);

            //liberamos lo que no vamos a usar mas esta iteraicion y repetimos si el tamaño del resultado es mayor que 10
            cudaFree(resultadoReductor);
            cudaFree(d_nuevosDatos);
        

        }


        //si hemos llegado aquie es que el tamaño del resultado es menor que 10, vamos a ver que hacemos

        //tomamos como resultado la primera posicion del vector que sabemos que siempre existira
        int resultadoFinal = resultadoReduccionVector[0];


        //recorremos el vector paso a paso y comparamos el resultado con la posicion siguiente a la que estamos, si es mayor la dejamos en el resultado
        for (int i = 0; i < resultadoReduccionVector.size(); i++) {

            if (i < resultadoReduccionVector.size() - 1 ) {
            
                resultadoFinal = max(resultadoFinal, resultadoReduccionVector[i + 1]);

            }

        }

        //imprimimos el resultado
        printf("\n[Maximizacion Reduccion] %s %d\n", columna, resultadoFinal);

    }
    else {


        //este es identico al de arriba cambiando que kernels lanzamos y que el valor inicial es el maximo porque es el neutro en una minimizacion

        int valorInicial = INT_MAX;
        

        cudaMemcpy(resultado, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
        reductorMinimalSimple <<<blocksInGrid, threadsInBlock>>>(d_vectorDatos, resultado, tamanno);
        cudaDeviceSynchronize();
        cudaMemcpy(&resultadoAImprimir, resultado, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n[Minimizacion Simple] %s %d\n", columna, resultadoAImprimir);


        cudaMemcpy(resultado, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
        reductorMinimalBasico <<<blocksInGrid, threadsInBlock, threadsInBlock.x*sizeof(int)>>> (d_vectorDatos, resultado, tamanno);
        cudaDeviceSynchronize();
        cudaMemcpy(&resultadoAImprimir, resultado, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n[Minimizacion Basica] %s %d\n", columna, resultadoAImprimir);


        cudaMemcpy(resultado, &valorInicial, sizeof(int), cudaMemcpyHostToDevice);
        reductorMinimalIntermedio <<<blocksInGrid, threadsInBlock, threadsInBlock.x * sizeof(int) >>> (d_vectorDatos, resultado, tamanno);
        cudaDeviceSynchronize();
        cudaMemcpy(&resultadoAImprimir, resultado, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n[Minimizacion Intermedia] %s %d\n", columna, resultadoAImprimir);



        


        int* arrInicial = new int[blocksInGrid.x];
        for (int i = 0; i < blocksInGrid.x; i++) arrInicial[i] = valorInicial;

        cudaMemcpy(resultadoReductor, arrInicial, blocksInGrid.x * sizeof(int), cudaMemcpyHostToDevice);
        reductorMinimalReductor << <blocksInGrid, threadsInBlock, threadsInBlock.x * sizeof(int) >> > (d_vectorDatos, resultadoReductor, tamanno);
        cudaDeviceSynchronize();
        resultadoReduccionVector.resize(blocksInGrid.x);
        cudaMemcpy(resultadoReduccionVector.data(), resultadoReductor, blocksInGrid.x * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(resultadoReductor);

        while (resultadoReduccionVector.size() > 10) {

            dim3 blocksInGrid;
            dim3 threadsInBlock;
            size_t espacio = resultadoReduccionVector.size() * sizeof(int);
            int tamanno = resultadoReduccionVector.size();

            configurarKernel(espacio, blocksInGrid, threadsInBlock);

            cudaMalloc(&resultadoReductor, blocksInGrid.x * sizeof(int));
            cudaMemcpy(resultadoReductor, &valorInicial, blocksInGrid.x * sizeof(int), cudaMemcpyHostToDevice);

            int* d_nuevosDatos;
            cudaMalloc(&d_nuevosDatos, espacio);
            cudaMemcpy(d_nuevosDatos, resultadoReduccionVector.data(), espacio, cudaMemcpyHostToDevice);


            reductorMinimalReductor << < blocksInGrid, threadsInBlock, threadsInBlock.x * sizeof(int) >> > (d_nuevosDatos, resultadoReductor, tamanno);
            cudaDeviceSynchronize();

            resultadoReduccionVector.resize(blocksInGrid.x);
            cudaMemcpy(resultadoReduccionVector.data(), resultadoReductor, blocksInGrid.x * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(resultadoReductor);
            cudaFree(d_nuevosDatos);


        }


        int resultadoFinal = resultadoReduccionVector[0];

        for (int i = 0; i < resultadoReduccionVector.size(); i++) {

            if (i < resultadoReduccionVector.size() - 1) {

                resultadoFinal = min(resultadoFinal, resultadoReduccionVector[i + 1]);

            }

        }


        printf("\n[Minimacion Reduccion] %s %d\n", columna, resultadoFinal);




    }

    cudaFree(d_vectorDatos);
    cudaFree(resultado);
    

}








__global__ void contarOcurrencias(int* datos, int tamannoSolucion, int tamannoDatos, int* resultado) {


    //aqui inicalizamos la memoria compartida
    extern __shared__ int datosEnBloque[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    
    //este es un poco raro, pero para inicializar toda la memoria en vez de que lo haga un unico hilo lo van a hacer todos.
    //como sabemos que no se solapan? porque cada uno solo hace su posicion + los multiplos del tamaño de bloque. ninguno llega a coincidir y todas posiciones se cubren
    //inicializamos a 0 para poder sumar de uno en uno coincidencias
    for (int i = threadIdx.x; i < tamannoSolucion; i = i + blockDim.x) {

        datosEnBloque[i] = 0;

    }
    
    
    
    __syncthreads();

    
    //si el idx del hilo es menor que el tamaño de datos miras que hay en datos[idx]
    //esa informacion la usas de posicion para sumar uno atomicamente a la posicion de la memoria compartida que dicen los datos
    if (idx < tamannoDatos){
    

        atomicAdd(&datosEnBloque[datos[idx]], 1);
    
    }
    

    __syncthreads();

    //cuando todos los hilos han acabado de hacer el histograma local pasamos los datos al resultado
    //igual que antes cada hilo mira su posicion y añade atomicamente los datos del bloque a la solucion 
    //(en todos los histogramas locales la posicion 0 es el mismo id por ejemplo) por eso todos los hilos 0 de todos los bloques suman ahi
    //ademas el for incrementa con el tamaño de bloque para que dentro de un mismo bloque los hilos no se pisen
    for (int i = threadIdx.x; i < tamannoSolucion; i = i + blockDim.x) {


        atomicAdd(&resultado[i], datosEnBloque[i]);



    }


    //luego esperamos a todos los hilos y hemos terminado
    __syncthreads();
    


}




void lanzadorHistograma(int opcion1, int opcion2, vector<string>& originAirport, vector<int>& originID, vector<string>& destAirport, vector<int>& destID) {



    /*
    
    antes de empezar con el codigo vamos a hablar de la logica. el objetivo es contar ocurrencias. como lo vamos a hacer?

    vamos a mapear cada id con su string, todos, salida y llegada, luego vamos a mapear los ids con un entero unico. para que?
    vamos a suponer que tenemos estos ids "a, b, c, d, e, c, f" con esto no podemos trabajar bien, pero si cada uno fuese un numero entero
    tendriamos un mapa "a -> 0, b -> 1, c -> 2, d -> 3, e -> 4, f -> 5" y podemos transformar los ids en enteros (no necesariamente ordenados)
    para tener algo asi "0, 1, 2, 3, 4, 2, 5" vamos a trabajar entonces con estos datos. por que?

    porque cada hilo pued ver ese dato y asociarlo a una posicion unica de un array. entonces cada posicion esta linkeada con un unico id y el elemento de esa posicion puede ser lo que sea
    como por ejemplo la cantidad de veces que aparece. por que lo hacemos asi?

    porque en gpu no podemos trabajar con listas de tuplas. y aqui estamos linealizando una lista de tuplas con 2 mapas.
    
    luego cada bloque puede hacer que cada hilo se mire a si mismo y sume 1 a la posicion de los resultados en memoria compartida que coincida con su dato utilizando asi la GPU de manera eficiente a coste de mapear

    despues de tener histogramas parciales en memoria compartida la idea es juntarlos todos en el resultado global e imprimir lo que necesitemos

    TODO ESTO CON MATICES ADICIONALES QUE VEREMOS SEGUN MIREMOS EL CODIGO


    
    */



    //He comprobado que estan todos los datos asi que no va a haber problema con el bucle porque al tener todos el mismo tamaño no se va a desbordar por lado ninguno

    //aqui declaramos los dos primeros mapas (luego habra otro para desindexar), punteros para la GPU y el resultado que traeremos a CPU
    unordered_map<int, string> mapaIdsString;
    unordered_map<int, int> mapaIdsIndice;
    int* resultado;
    int* d_originID;
    int* d_destID;
   
    vector<int> resultadoKernel;



    //mapeamos los datos de ID a Codigo de aeropuerto
    int i = 0;

    while (i < originAirport.size()) {
    

        mapaIdsString.insert({ destID[i], destAirport[i] });
        mapaIdsString.insert({ originID[i], originAirport[i] });
     
        i++;

    
    }


    //mapeamos de ID a indice

    int j = 0;
    int k = 0;

    while (j < originAirport.size()) {
    
        if(mapaIdsIndice.count(destID[j]) == 0){
        
            mapaIdsIndice.insert({ destID[j], k });
            k++;

        }

        if (mapaIdsIndice.count(originID[j]) == 0) {

            mapaIdsIndice.insert({ originID[j], k });
            k++;

        }

        j++;
        
    
    }


    //configuramos el kernel

    dim3 blocksInGrid;
    dim3 threadsInBlock;
    size_t espacio = originID.size() * sizeof(int);

    configurarKernel(espacio, blocksInGrid, threadsInBlock);

    //alocamos espacio para el resultado. tiene ese tamaño porque vamos a tener una posicion por indice (equivalente a una posicion por id unico o aeropuerto)
    cudaMalloc(&resultado, mapaIdsIndice.size() * sizeof(int));
    

    //este if nos dice si usamos los de origen o los de destino, ambos tienen la misma logica
    if (opcion1 == 1) {

        //hacemos una copia de los datos antes de modificarlos porque si ejecutasemos esto mas de una vez sin volver a hacer carga los datos serian los indices y no queremos eso
        vector<int> originIDCopia = originID;

        //transformamos cada id a su indice usando el mapa, no hay proteccion ninguna porque sabemos que todo lo que busquemos esta en el mapa
        for (int i = 0; i < originID.size(); i++) {

            auto it = mapaIdsIndice.find(originID[i]);
            originIDCopia[i] = it->second;
            

        }

        //alocamos y compiamos la COPIA de los datos, si copiasemos los datos no habriamos hecho nada
        cudaMalloc(&d_originID, originID.size() * sizeof(int));
        cudaMemcpy(d_originID, originIDCopia.data(), originID.size() * sizeof(int), cudaMemcpyHostToDevice);

        //lanzamos el kernel con memoria compartida = numero de ids distintos
        contarOcurrencias<<<blocksInGrid, threadsInBlock, mapaIdsIndice.size() * sizeof(int) >>>(d_originID, mapaIdsIndice.size(), originID.size(), resultado);

        //luego despues liberamos lo que hemos usado aqui y no necesitamos
        cudaFree(d_originID);
    
    }
    else {


        vector<int> destIDCopia = destID;

        for (int i = 0; i < destID.size(); i++) {

            auto indice = mapaIdsIndice.find(destID[i]);
            destIDCopia[i] = indice->second;

        }

        cudaMalloc(&d_destID, destID.size() * sizeof(int));
        cudaMemcpy(d_destID, destIDCopia.data(), destID.size() * sizeof(int), cudaMemcpyHostToDevice);
     
        contarOcurrencias<<<blocksInGrid, threadsInBlock, mapaIdsIndice.size() * sizeof(int) >>>(d_destID, mapaIdsIndice.size(), destID.size(), resultado);

        cudaFree(d_destID);
    
    }


     
    //vamos a traernos los resultados del kernel al vector que teniamos para ello, haciendo el resize primero para darle espacio
    resultadoKernel.resize(mapaIdsIndice.size());
    cudaMemcpy(resultadoKernel.data(), resultado, mapaIdsIndice.size() * sizeof(int), cudaMemcpyDeviceToHost);


    //ahora tenemos un vector donde la posicion es el indice asociado al id y lo que hay en la posicion es el numero de ocurrencias
    //vamos a encontrar el elemento que tiene el maximo
    auto it = max_element(resultadoKernel.begin(), resultadoKernel.end());

    //y copiamos el maximo aqui, le vamos a usar para decidir cuantos # le ponemos a cada aeropuerto cuando lo mostremos
    int max = *it;

    //definimos esto para usarlo en el siguiente while, ahora lo vemos
    int siguienteMaximo = max;
    int posicion = it - resultadoKernel.begin();


    //aqui tenemos que mapear al reves, indices a ids para poder buscar los ids mas adelante
    unordered_map<int, int> mapaIndiceIds;

    for (auto it = mapaIdsIndice.begin(); it != mapaIdsIndice.end(); ++it) {

        mapaIndiceIds.insert({ it->second, it->first });
        
    }

    

    //aqui es donde se hace el histograma como tal, la opcion2 era la cota que nos daba el usuario entonces no vamos a mostrar cosas menores

    while (siguienteMaximo >= opcion2) {
    
        //tomamos el id y el codigo del aeropuerto del maximo que estamos mirando e imprimimos
        int id = mapaIndiceIds.find(posicion)->second;
        string codAeropuerto = mapaIdsString.find(id)->second;
    
        cout << "El aeropuerto " << codAeropuerto << " con id " << id << " aparece " << siguienteMaximo << " veces. ";


        int totalAsteriscos = 15;
        //por que esta conversion es critica?
        //si dividiesemos normal int/int siempre da entero y como menos el primer maximo todo es < 1 siempre tendriamos 0 asteriscos
        //al hacer la conversion dejamos los decimales antes de multiplicar, ademas como se va a truncar en el for siguiente le añadimos 0.5f para que redondee, no trunque
        float asteriscos = (((float)siguienteMaximo / (float)max) * totalAsteriscos) + 0.5f;


        //aqui imprimimos tantos asteriscos como hemos calculado
        for (int i = asteriscos; i > 0; i--) {


            cout << "#";


        }


        cout << "\n";


        //importante esta linea porque si no el maximo se repetiria todo el rato si no lo quitamos.
        //por que -1? porque el usuario puede pedir el 0 de umbral pero no nada negativo. si lo pusiesemos a 0 llegaria un punto donde todo serian maximos y se repetiria indefinidamente

        resultadoKernel[posicion] = -1;



        //ahora seleccionamos otra maximo nuevo

        it = max_element(resultadoKernel.begin(), resultadoKernel.end());

        siguienteMaximo = *it;
        posicion = it - resultadoKernel.begin();

    
    }

    //para saber que hemos ternminado decimos esto y liberamos resultado
    cout << "Fin de resultados.\n";

    cudaFree(resultado);
    

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
    vector<float> weatherDelay;
    vector<float> arrTime;
    vector<float> depTime;
    vector<string> tailNum;
    vector<string> originAirport;
    vector<string> destAirport;
    vector<int> originID;
    vector<int> destID;

    int limite = 0;  //Cambiar para cargar mas o menos datos (0 para cargarlos todos)

    if (ruta == "") {
        cout << "\nCargando con ruta por defecto\n";
        //FUNCION DE CARGA CON LA RUTA POR DEFECTO
        //Ruta por defecto Jose Antonio:
        //leerCSV("D:/Fichero PAP/Airline_dataset.csv", arrDelay, depDelay, tailNum, limite);
        //Ruta por defecto Jorge:
        leerCSV("C:/Users/Jorge/Documents/Airline_dataset.csv", arrDelay, depDelay, weatherDelay, arrTime, depTime, tailNum, originAirport, destAirport, originID, destID, limite);

    }
    else
    {
        cout << "\nCargando con ruta: " << ruta << "\n" << endl;
        //FUNCION DE CARGA CON LA RUTA ESPECIFICADA
        leerCSV(ruta, arrDelay, depDelay, weatherDelay, arrTime, depTime, tailNum, originAirport, destAirport, originID, destID, limite);
    }


    int opcion;
    bool ejecutar = true;

    while (ejecutar) {
        cout << "\n--- MENU DE OPCIONES ---\n";
        cout << "1. Retraso en despegues\n";
        cout << "2. Retraso en aterrizajes\n";
        cout << "3. Reduccion de retraso\n";
        cout << "4. Histograma de aeropuertos\n";
        cout << "5. Salir\n\n";
        cout << "Elija la opcion: ";
        cin >> opcion;
        cout << "\n";

        //Switch case sencillo para manejar las 5 opciones posibles
        switch (opcion) {
        case 1: {

            float umbral;
            bool opcionNoValida = true;
            
            while (opcionNoValida) {


                cout << "Introduzca el umbral (positivo para retrasos, negativo para adelantos): ";
                cin >> umbral;


                if (!cin) { //Que me han dado un float bien, que no salgo y reinicio el cin

                    cout << "\nOpcion no valida\n";
                    //Para el caso de que no pusiera un numero
                    cin.clear(); //Limpia errores
                    cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
                    continue;

                }

                opcionNoValida = false;

            
            }


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

            bool opcionNoValida = true;
            float umbral;
            cout << "Introduzca el umbral (positivo para retrasos, negativo para adelantos): ";
            cin >> umbral;

            while (opcionNoValida) {
            
                if (!cin) { //Que me han dado un float bien, que no salgo y reinicio el cin

                    cout << "\nOpcion no valida\n";
                    //Para el caso de que no pusiera un numero
                    cin.clear(); //Limpia errores
                    cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
                    continue;

                }

                opcionNoValida = false;
            
            }

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


            //En este case vamos a hacer el ejercicio 3. El objetivo de este trozo de codigo es inicializar un lanzador en CPU que nos lanzara el kernel.


            //Definimos variables que usaremos
            int selector1;
            int selector2;
            bool opcionInvalida = true;

            while (opcionInvalida) {
           
                //Le damos al usuario las opciones, hasta que no elija una valida o salga este bucle se repetira por la variable opcionInvalida
                cout << "\n---SELECTOR DE REDUCCIONES---\n";
                cout << "1) Retraso de Salida.\n";
                cout << "2) Retraso de Llegada.\n";
                cout << "3) Retraso por el Tiempo.\n";
                cout << "4) Horas de salida.\n";
                cout << "5) Horas de llegada.\n";
                cout << "6) Salir.\n\n";


  
                cin >> selector1; //Elijo opcion
                cout << "\n";

                //validamos lo que nos ha pedido el usario
                if (!cin) { //Si he elegido un entero me salto esto, si no limpiamos el cin y reiniciamos el bucle

                    cout << "\nOpcion no valida\n";
                    //Para el caso de que no pusiera un numero
                    cin.clear(); //Limpia errores
                    cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
                    continue;

                }

                if (selector1 == 6) { //Compruebo si me han dicho que quieren salir

                    opcionInvalida = false;
                    break;

                }




                //Le preguntamos al usuario que operacion quiere hacer
                cout << "1) Maximizar.\n";
                cout << "2) Minimizar.\n";
                cout << "3) Salir.\n\n";

                
                
                cin >> selector2; //Pregunto que quiere el usuario
                cout << "\n";

                //validamos que nos haya pasado un entero
                if (!cin) { //Que me han dado un entero bien, que no salgo y reinicio el cin

                    cout << "\nOpcion no valida\n";
                    //Para el caso de que no pusiera un numero
                    cin.clear(); //Limpia errores
                    cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
                    continue;

                }


                if (selector2 == 3) { //Que me piden salir, salgo

                    opcionInvalida = false;
                    break;

                }
                
                
                //validamos si el entero es valido, no nos interesan negaticos ni mayores que 6 para la primera opcion ni cosas distintas de 1 o 2 en la segunda (si pidio salir ya salio)
                if ((selector1 > 0 && selector1 < 6) && (selector2 == 1 || selector2 == 2)) { //Que la opcion es una de las del selector, perfecto, que no, salgo.

                    //lanzamos el lanzador y salimos del while
                    lanzadorReductor(selector1, selector2, depDelay, arrDelay, weatherDelay, depTime, arrTime);
                    
                    opcionInvalida = false;
                    break;

                }
                else {
                
                    cout << "\nOpcion no valida\n";
                
                }
                
                
                
              
            }


            break;
        }

        case 4: {
            
            //El obejtivo de este trocito es inicializar el lanzador con los datos que necesite y hacer de interfaz para el usuario

            //aqui declaro lo que voy a ir usando
            int selector;
            int selector2;
            bool opcionInvalida = true;


            //es un bucle muy sencillo que simplemente le pide al usario que elija entre las 3 y comprueba que la opcion sea valida, lo mismo hace el siguiente bucle
            while (opcionInvalida) {
            


                cout << "1) Histograma aeropuertos de salida\n";
                cout << "2) Histograma aeropuertos de llegada\n";
                cout << "3) Salir\n\n";

                cin >> selector;

                cout << "\n";


                if (!cin || selector < 0 || selector > 3) { //Que me han dado un entero bien, que no salgo y reinicio el cin

                    cout << "\nOpcion no valida\n";
                    //Para el caso de que no pusiera un numero
                    cin.clear(); //Limpia errores
                    cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
                    continue;

                }

                break;

            
            }
            

            //aqui si se ha elegido la 3 antes

            if (selector == 3) { //Que me piden salir, salgo

                opcionInvalida = false;
                break;

            }

            while (opcionInvalida) {



                cout << "Introduzca un umbral para mostrar resultados.\n\n";

                cin >> selector2;

                cout << "\n";


                if (!cin || selector2 < 0) { //Que me han dado un entero bien, que no salgo y reinicio el cin

                    cout << "\nOpcion no valida\n";
                    //Para el caso de que no pusiera un numero
                    cin.clear(); //Limpia errores
                    cin.ignore(numeric_limits<streamsize>::max(), '\n'); //Descarta la linea que ha introducido por consola,da igual como de larga sea
                    continue;

                }


                break;


            }


            //y ya lanzamos el lanzador, la logica de este ejercicio mas adelante
            
            lanzadorHistograma(selector, selector2, originAirport, originID, destAirport, destID);


            break;
        }

        case 5:
            cout << "\nSaliendo...\n";
            ejecutar = false;   //Terminamos el bucle
            break;


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
