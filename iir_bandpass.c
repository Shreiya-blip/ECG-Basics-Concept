#include <stdio.h>

#define MAX_SAMPLES 10800
#define SECTIONS 4

int main()
{
    FILE *input, *coeff, *output;

    double ecg[MAX_SAMPLES];
    double sos[SECTIONS][6];

    double z1[SECTIONS] = {0};
    double z2[SECTIONS] = {0};

    int count = 0;

    // Step 1: Read ECG input samples
    input = fopen("ecg_input.txt", "r");

    if (input == NULL)
    {
        printf("Error opening ECG input file!\n");
        return 1;
    }

    while (count < MAX_SAMPLES &&
           fscanf(input, "%lf", &ecg[count]) == 1)
    {
        count++;
    }

    fclose(input);

    printf("ECG samples loaded: %d\n", count);

    // Step 2: Read SOS coefficients
    coeff = fopen("sos_coefficients.txt", "r");

    if (coeff == NULL)
    {
        printf("Error opening coefficient file!\n");
        return 1;
    }

    for (int i = 0; i < SECTIONS; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            if (fscanf(coeff, "%lf", &sos[i][j]) != 1)
            {
                printf("Error reading coefficients!\n");
                fclose(coeff);
                return 1;
            }
        }
    }

    fclose(coeff);

    // Step 3: Apply IIR bandpass filter
    for (int n = 0; n < count; n++)
    {
        double x = ecg[n];

        for (int s = 0; s < SECTIONS; s++)
        {
            double b0 = sos[s][0];
            double b1 = sos[s][1];
            double b2 = sos[s][2];

            double a0 = sos[s][3];
            double a1 = sos[s][4];
            double a2 = sos[s][5];

            // Normalize coefficients by a0
            double y = (b0 / a0) * x + z1[s];

            double new_z1 =
                (b1 / a0) * x - (a1 / a0) * y + z2[s];

            double new_z2 =
                (b2 / a0) * x - (a2 / a0) * y;

            z1[s] = new_z1;
            z2[s] = new_z2;

            x = y;
        }

        ecg[n] = x;
    }

    // Step 4: Save filtered ECG output
    output = fopen("c_bandpass_output.txt", "w");

    if (output == NULL)
    {
        printf("Error creating output file!\n");
        return 1;
    }

    for (int i = 0; i < count; i++)
    {
        fprintf(output, "%.17g\n", ecg[i]);
    }

    fclose(output);

    printf("IIR bandpass filtering completed!\n");
    printf("Filtered samples: %d\n", count);
    printf("Output saved to c_bandpass_output.txt\n");

    return 0;
}