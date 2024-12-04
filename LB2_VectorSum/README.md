# Перемножение матриц

- Задача: реализовать алгоритм сложения элементов вектора
- Язык: C++
- Входные данные: Вектор размером 1 000..1 000 000 значений.
- Выходные данные: сумма элементов вектора + время вычисления


## Реализация
[Файл с кодом программы](./CudaRuntime2/CudaRuntime2/kernel.cu)

В коды представлены двe основные функции
- `vectorSumWithCPU` - сложение вектора на CPU
- `vectorSumWithCUDA` - сложение вектора на GPU при помощи CUDA

Принцип расспараллеливания заключается в том, чтобы каждая нить считает сумму массива до тех пор, пока он не закончится, то есть, если нитей всео 100, а массив размером 1500, то каждая нить посчитает сумму 15 элементов. После чего полученное значение записывается в массив, который хранится в общей памяти (`__shared__`). На следующем шаге все элементы из общей памяти складываются так, что сумма оказывается в `[0]` и записывается в выходной массив. Так, прогнав алгоритм 2 раза, в первый раз на массиве, а второй на выходном массиве, можно получить сумму исходного массива.

Алгоритм работает с любой размерностью сетки и блоков, однако в данной работе использовались значения 12 и 1024 соответственно.

Код ядра можно разделить следующие этапы (ядро запускается дважды):
- сумма значений нитью, которые необходимо ей сложить
- запись полученных элементов в общую память
- редукция значений из общей памяти
- запись в out-массив значений каждого блока


## Таблица времени работы на CPU и GPU
В таблице представлены 

| Vector size (N) | Time CPU (msec) | Time GPU (msec) | Boost (CPU/GPU) |
|-----------------|-----------------|-----------------|-----------------|
| 1_000           | 0.05            | 0.26            | 0.19            |
| 10_000          | 0.05            | 0.24            | 0.21            |
| 100_000         | 0.95            | 0.22            | 4.31            |
| 1_000_000       | 6.05            | 0.35            | 17.28           |
| 10_000_000      | 69.43           | 1.17            | 59.34           |

### Описание
- **Vector size (N x N)**: Размеры векторов, для которых проводятся измерения.
- **Time CPU (сек)**: Усредненное время выполнения операции на центральном процессоре по 5 запускам.
- **Time GPU (сек)**: Усредненное время выполнения операции на графическом процессоре по 5 запускам.
- **Boost (CPU/GPU)**: Ускорение, вычисление производилось как отношение времени на CPU к времени на GPU.

##
Замеры произоводились на следующей системе:
- **CPU:** Intel(R) Pentium(R) CPU G4600 @ 3.60GHz
- **GPU:** Nvidia GeForce®GTX 1050 Ti 4gb
- **RAM:** 16gb, 2400MHz
