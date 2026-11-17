#include <stdio.h>

#define MAX_SAMPLES 10800

int main()
{
    FILE *fp;
    double ecg[MAX_SAMPLES];
    int count = 0;

    fp = fopen("ecg_input.txt", "r");

    if (fp == NULL)
    {
        printf("Cannot open ECG file!\n");
        return 1;
    }

    while (count < MAX_SAMPLES &&
           fscanf(fp, "%lf", &ecg[count]) == 1)
    {
        count++;
    }

    fclose(fp);

    printf("ECG file loaded successfully!\n");
    printf("Number of samples: %d\n", count);

    printf("\nFirst 10 samples:\n");

    for (int i = 0; i < 10 && i < count; i++)
    {
        printf("%d : %.10f\n", i, ecg[i]);
    }

    return 0;
}