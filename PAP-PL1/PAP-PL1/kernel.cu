
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;


int main()
{

    string ruta = "";

    cout << "\nEL1 PAP 2026 Jorge Lopez y Jose Antonio Lopez\n";
    cout << "Introduzca la ruta base del dataset (o pulse intro para usar C:\NOMBRE_DEL_DATASET.csv): ";
    getline(cin, ruta);

    if (ruta == "") {
        cout << "\nCargando con ruta por defecto\n";
        //FUNCION DE CARGA CON LA RUTA POR DEFECTO
    }
    else
    {
        cout << "\nCargando con ruta: " << ruta << "\n" << endl;
        //FUNCION DE CARGA CON LA RUTA POR DEFECTO
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
