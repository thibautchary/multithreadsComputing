// Compilation : gcc -Wall -std=c99 -O3 -march=native -mtune=native -o info info.c -lm -lpthread
// Exécution   : ./info

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <unistd.h>

// Librairie pour les threads.
#include <pthread.h>

// Intrinsics du compilateur pour la vectorisation.
#include <immintrin.h>

// Structure pour les arguments des fonctions threads.
typedef struct {
    float *u;
    float *v;
    int begin;
    int end;
    float sum;
} thread_args_t;

// Fonctions threads.
void *thread_fctn (void *args) {
    // Argument de la fonction threads.
    thread_args_t *copy = (thread_args_t *) args;

    // Somme partielle de cette fonction.
    float sum = 0.0f;

#if defined (__AVX__)
    // Si AVX est disponible, utilise des vecteurs de 8 floats.
    for (int i = copy->begin; i < copy->end; i += 8) {
        __m256 y1, y2, y3;

        // Copie les vecteurs dans les registres.
        y1 = _mm256_load_ps (&copy->u [i]);
        y2 = _mm256_load_ps (&copy->v [i]);

        // Effectue les racines carrées.
        y1 = _mm256_sqrt_ps (y1);
        y2 = _mm256_sqrt_ps (y2);

        // Effecteur la soustraction membre à membre.
        y3 = _mm256_sub_ps (y1, y2);

        // Élève le résultat, membre à membre, au carré.
        y1 = _mm256_mul_ps (y3, y3);

        // Effectue l'addition horizontale des 8 composantes
        __m128 x1, x2, x3;
        // (x7, x6, x5, x4)
        x1 = _mm256_extractf128_ps (y1, 1);
        // (x3, x2, x1, x0)
        x2 = _mm256_castps256_ps128 (y1);
        // (x3 + x7, x2 + x6, x1 + x5, x0 + x4)
        x1 = _mm_add_ps (x2, x1);
        // (-, -, x3 + x7, x2 + x6)
        x2 = _mm_movehl_ps (x1, x1);
        // (-, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6)
        x3 = _mm_add_ps (x1, x2);
        // (-, -, -, x1 + x3 + x5 + x7)
        x2 = _mm_shuffle_ps (x3, x3, 0x1);
        // (-, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7)
        x1 = _mm_add_ss (x3, x2);

        sum += _mm_cvtss_f32 (x1);
    }
#elif defined (__SSE__)
    // Si SSE est disponible, utilise des vecteurs de 4 floats.
    for (int i = copy->begin; i < copy->end; i += 4) {
        __m128 x1, x2, x3;

        // Copie les vecteurs dans les registres.
        x1 = _mm_load_ps (&copy->u [i]);
        x2 = _mm_load_ps (&copy->v [i]);

        // Effectue les racines carrées.
        x1 = _mm_sqrt_ps (x1);
        x2 = _mm_sqrt_ps (x2);

        // Effecteur la soustraction membre à membre.
        x3 = _mm_sub_ps (x1, x2);

        // Élève le résultat, membre à membre, au carré.
        x1 = _mm_mul_ps (x3, x3);

        // Effectue l'addition horizontale des 4 composantes
#if defined (__SSE3__)
        // (x2 + x3, x0 + x1, x2 + x3, x0 + x1)
        x1 = _mm_hadd_ps (x1, x1);
        // (x1 + x2 + x3 + x4, x1 + x2 + x3 + x4, x1 + x2 + x3 + x4, x1 + x2 + x3 + x4)
        x1 = _mm_hadd_ps (x1, x1);
#else
        // (-, -, x1, x2)
        x2 = _mm_movehl_ps (x1, x1);
        // (-, -, x1 + x3, x2 + x4)
        x2 = _mm_add_ps (x1, x2);
        // (-, -, -, x1 + x3)
        x3 = _mm_shuffle_ps (x2, x2, 0x1);
        // (-, -, -, x1 + x2 + x3 + x4)
        x1 = _mm_add_ss (x2, x3);
#endif

        sum += _mm_cvtss_f32 (x1);
    }
#else
    // Autrement, utilise l'algorithme de base.
    for (int i = copy->begin; i < copy->end; i++) {
        float tmp = (sqrtf (copy->u [i]) - sqrtf (copy->v [i]));
        sum += tmp * tmp;
    }
#endif

    // Stock la somme partielle.
    copy->sum = sum;

    pthread_exit (0);
}

// Fonction de distance non optimisée, qui utilise l'algorithme de base.
float distance (float *u, float *v, int n) {
    float d = 0.0f;

    for (int i = 0; i < n; i++) {
        float tmp = (sqrtf (u [i]) - sqrtf (v [i]));
        d += tmp * tmp;
    }

    return d;
}

// Fonction de distance optimisée, qui utilise des threads.
float distance_opt (float *u, float *v, int n, int nprocs) {
    // Tableaux pour les threads et leurs arguments.
    pthread_t *threads = (pthread_t *) malloc (sizeof (pthread_t) * nprocs);
    thread_args_t **args = (thread_args_t **) malloc (sizeof (thread_args_t *) * nprocs);

    // Distance.
    float d = 0.0f;

    // Définit la taille des vecteurs.
#if defined (__AVX__)
    int size = 8;
#elif defined (__SSE__)
    int size = 4;
#else
    int size = 1;
#endif

    // Calcul l'intervalle de calcul vectoriel (doit être multiple de size).
    int end = n - (n % size);

    // Calcul, en arrondissant, l'intervalle de calcul pour les nprocs - 1
    // premiers threads. Le dernier thread effectue le calcul jusqu'à "end".
    int k = end / nprocs;

    // Somme sur le nombre de threads à créer.
    for (int i = 0; i < nprocs; i++) {
        // Alloue la mémoire pour les arguments.
        args [i] = (thread_args_t *) malloc (sizeof (thread_args_t));

        // Copie les pointeurs vers les vecteurs, et définit l'intervalle
        // de calcul.
        args [i]->u = u;
        args [i]->v = v;
        args [i]->begin = i * k;
        args [i]->end = (i + 1) * k;

        // Pour le dernier thread, effectue le calcul jusqu'à "end", légèrement
        // plus que les autres.
        if ((i + 2) * k > end) {
            args [i]->end = end;
        }

        // Créé le thread.
        pthread_create (&threads [i], NULL, &thread_fctn, (void *) args [i]);
    }

    // Somme sur le nombre de threads créés.
    for (int i = 0; i < nprocs; i++) {
        // Attend la fin du thread.
        pthread_join (threads [i], NULL);

        // Récupère la somme partielle, et l'ajoute à la distance.
        d += args [i]->sum;

        // Libère les arguments.
        free (args [i]);
    }

    // Effectue la fin du calcul, qui n'a pas pu être vectorisée (partie de n
    // non multiple de size).
    for (int i = end; i < n; i++) {
        float tmp = (sqrtf (u [i]) - sqrtf (v [i]));
        d += tmp * tmp;
    }

    // Libère la mémoire des threads et de leurs arguments.
    free (threads);
    free (args);

    return d;
}

// Initialize les vecteurs.
void init (float *u, float *v, int n) {
    srand (time (NULL));

    // Rempli les vecteurs avec des flotants aléatoires.
    for (int i = 0; i < n; i++) {
        // rand () retourne un entier entre 0 et RAND_MAX.
        u [i] = (float) rand () / RAND_MAX;
        v [i] = (float) rand () / RAND_MAX;
    }

    return;
}

int main (void) {
    // Taille des vecteurs
    int n = 10000000;

    // Alloue la mémoire pour les deux vecteurs, alignée sur 16 bits.
    float *u, *v;
    posix_memalign ((void **) &u, 16, sizeof (float) * n);
    posix_memalign ((void **) &v, 16, sizeof (float) * n);

    // Initialize les vecteurs
    init (u, v, n);

    // Variables pour mesurer le temps de calcul
    struct timeval start, end;

    // Calcul le nombre de processeurs, pour déterminer le nombre de threads
    // à démarrer.
    int nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs < 1) {
        nprocs = 1;
    }
    printf ("Utilisation de %i threads.\n", nprocs);
#if defined (__AVX__)
    printf ("Vectorisation avec AVX\n");
#elif defined (__SSE__)
  #if defined (__SSE3__)
    printf ("Vectorisation avec SSE3\n");
  #else
    printf ("Vectorisation avec SSE\n");
  #endif
#else
    printf ("Aucune vectorisation.\n");
#endif

    printf ("\n");

    float d1, d2, e1, e2;

    // Calcule la distance, version "simple"
    gettimeofday (&start, NULL);
    d1 = distance (u, v, n);
    gettimeofday (&end, NULL);

    e1 = (float) ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000000.0f;
    printf ("distance = %.*f,\tcalculée en %.3f s\n", FLT_DIG, d1, e1);

    // Calcule la distance, version "optimisée"
    gettimeofday (&start, NULL);
    d2 = distance_opt (u, v, n, nprocs);
    gettimeofday (&end, NULL);

    e2 = (float) ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000000.0f;
    printf ("distance = %.*f,\tcalculée en %.3f s\n", FLT_DIG, d2, e2);

    printf ("\nFacteur d'accélération : %.3f\n", e1 / e2);

    // Libère la mémoire allouée pour les vecteurs.
    free (u);
    free (v);

    return 0;
}

