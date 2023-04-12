Make file modificado para que funcione en el portatil
he añadido a libs las librerias -lcuda y -lcudart
El profe a puesto otro makefile ya corregido pero de momento me quedo con el mio

Revisar el canny que claramente no va, lo que se puede revisar son los parametros de entrada a funciones, no vaya a ser que el copilot me la haya liado y prbar 1 por 1 los kernels que he hecho para ver cual es el puto.

Efectivamente el copilot la lio con las alturas y las anchuras, ahora hay que revisar una violacion de segmento a la hora de lanzar el kernel de hough, vease revisar que haya reservado toda la memoria necesaria para hacer los calculos.

Pues no se porque me dio la violacion de segmento, pero de todas formas ya esta todo en la gpu ahora hay que optimizar lo maximo posible, lo mas posible es que el getlines este pocho y hay que mirar lo de la memoria compartida.