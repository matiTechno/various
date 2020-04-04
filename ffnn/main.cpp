#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <arpa/inet.h>

struct Net
{
    int out_size;
    int hid_size;
    int in_size;

    float* out_a; // activation values, neurons
    float* hid_a;

    float* ho_weights; // hidden to output
    float* ih_weights; // input to hidden

    float* out_biases;
    float* hid_biases;

    // data for training

    float* out_z; // a = fn_activation(z), z is called a weighted input or sum
    float* hid_z;

    // gradient_sum represents a sum of gradients; for every training example in a batch a gradient is calculated and
    // added to the sum, later this sum is divided by a batch size to obtain an average gradient

    float* gradient_sum_ho_weights; // cost function gradient with respect to weights (hidden to output)
    float* gradient_sum_ih_weights;

    float* gradient_sum_out_biases;
    float* gradient_sum_hid_biases;

    float* gradient_out_z; // often symbolized as delta
};

// couldn't find a better name...

struct Train
{
    int epoch_count;
    int batch_size;
    float eta; // gradient step size multiplier, also known as a learning rate

    // note, input_set[n] and output_set[n] form a single training example
    int set_size;
    float** input_set;
    float** output_set;

    // this set is used only for testing (evaluation) and should not have common parts with the training set
    int test_set_size;
    float** test_input_set;
    float** test_output_set;
};

std::mt19937 _rng;

float get_random01()
{
    std::uniform_real_distribution<float> d(0.f, 1.f);
    return d(_rng);
}

void net_init(Net& net, int in_size, int hid_size, int out_size)
{
    net.in_size = in_size;
    net.hid_size = hid_size;
    net.out_size = out_size;

    int fs = sizeof(float);

    net.out_a = (float*)malloc(fs * out_size);
    net.hid_a = (float*)malloc(fs * hid_size);

    net.ho_weights = (float*)malloc(fs * out_size * hid_size);
    net.ih_weights = (float*)malloc(fs * hid_size * in_size);

    net.out_biases = (float*)malloc(fs * out_size);
    net.hid_biases = (float*)malloc(fs * hid_size);

    net.out_z = (float*)malloc(fs * out_size);
    net.hid_z = (float*)malloc(fs * hid_size);

    net.gradient_sum_ho_weights = (float*)malloc(fs * out_size * hid_size);
    net.gradient_sum_ih_weights = (float*)malloc(fs * hid_size * in_size);

    net.gradient_sum_out_biases = (float*)malloc(fs * out_size);
    net.gradient_sum_hid_biases = (float*)malloc(fs * hid_size);

    net.gradient_out_z = (float*)malloc(fs * out_size);

    for(int i = 0; i < out_size * hid_size; ++i)
        net.ho_weights[i] = get_random01() * 2.f - 1.f;

    for(int i = 0; i < hid_size * in_size; ++i)
        net.ih_weights[i] = get_random01() *  2.f - 1.f;

    for(int i = 0; i < out_size; ++i)
        net.out_biases[i] = get_random01() * 2.f - 1.f;

    for(int i = 0; i < hid_size; ++i)
        net.hid_biases[i] = get_random01() * 2.f - 1.f;
}

float fn_activation(float z)
{
    return 1.f / (1.f + expf(-z)); // sigmoid
}

// derivative

float fn_activation_dr(float z)
{
    return fn_activation(z) * (1.f - fn_activation(z));
}

void net_fprop(Net net, float* input)
{
    // input to hidden

    for(int i = 0; i < net.hid_size; ++i)
    {
        float z = net.hid_biases[i];

        for(int k = 0; k < net.in_size; ++k)
            z += net.ih_weights[i * net.in_size + k] * input[k];

        net.hid_z[i] = z;
        net.hid_a[i] = fn_activation(z);
    }

    // hidden to output

    for(int i = 0; i < net.out_size; ++i)
    {
        float z = net.out_biases[i];

        for(int k = 0; k < net.hid_size; ++k)
            z += net.ho_weights[i * net.hid_size + k] * net.hid_a[k];

        net.out_z[i] = z;
        net.out_a[i] = fn_activation(z);
    }
}

void backprop(Net net, float* input, float* target) // computes gradients
{
    // output layer

    for(int i = 0; i < net.out_size; ++i)
    {
        float dCdz = fn_activation_dr(net.out_z[i]) * (net.out_a[i] - target[i]); // partial derivative of C with respect to z, C is a cost function
        net.gradient_out_z[i] = dCdz; // this is used by a hidden layer
        net.gradient_sum_out_biases[i] += dCdz; // see comments why +=

        for(int k = 0; k < net.hid_size; ++k)
            net.gradient_sum_ho_weights[i * net.hid_size + k] += net.hid_a[k] * dCdz;
    }

    // hidden layer

    for(int i = 0; i < net.hid_size; ++i)
    {
        float sum = 0;

        for(int k = 0; k < net.out_size; ++k)
            sum += net.ho_weights[k * net.hid_size + i] * net.gradient_out_z[k];

        float dCdz = fn_activation_dr(net.hid_z[i]) * sum;
        net.gradient_sum_hid_biases[i] += dCdz;

        for(int k = 0; k < net.in_size; ++k)
            net.gradient_sum_ih_weights[i * net.in_size + k] += input[k] * dCdz;
    }
}

void shuffle_set(float** input_set, float** output_set, int set_size)
{
    for(int i = 0; i < set_size - 2; ++i)
    {
        std::uniform_int_distribution<int> d(i, set_size - 1);
        int j = d(_rng);

        float* tmp = input_set[i];
        input_set[i] = input_set[j];
        input_set[j] = tmp;

        tmp = output_set[i];
        output_set[i] = output_set[j];
        output_set[j] = tmp;
    }
}

void net_train(Net net, Train train)
{
    for(int _i = 0; _i < train.epoch_count; ++_i)
    {
        shuffle_set(train.input_set, train.output_set, train.set_size);

        for(int k = 0; k < train.set_size; k += train.batch_size) // for each batch
        {
            if(k + train.batch_size > train.set_size)
                break;

            // zero out gradient sums

            int fs = sizeof(float);
            memset(net.gradient_sum_ho_weights, 0, fs * net.out_size * net.hid_size);
            memset(net.gradient_sum_ih_weights, 0, fs * net.hid_size * net.in_size);
            memset(net.gradient_sum_out_biases, 0, fs * net.out_size);
            memset(net.gradient_sum_hid_biases, 0, fs * net.hid_size);

            // for each training example in the batch calculate a gradient and add it to the previous gradients

            for(int i = 0; i < train.batch_size; ++i)
            {
                float* input = train.input_set[k + i];
                float* target = train.output_set[k + i];
                net_fprop(net, input);
                backprop(net, input, target);
            }

            // apply stochastic gradient descent to weights and biases

            float step_coeff = train.eta / train.batch_size;

            for(int i = 0; i < net.out_size * net.hid_size; ++i)
                net.ho_weights[i] -= step_coeff * net.gradient_sum_ho_weights[i];

            for(int i = 0; i < net.hid_size * net.in_size; ++i)
                net.ih_weights[i] -= step_coeff * net.gradient_sum_ih_weights[i];

            for(int i = 0; i < net.out_size; ++i)
                net.out_biases[i] -= step_coeff * net.gradient_sum_out_biases[i];

            for(int i = 0; i < net.hid_size; ++i)
                net.hid_biases[i] -= step_coeff * net.gradient_sum_hid_biases[i];
        }

        // evaluate new weights and biases

        int match_count = 0;

        for(int k = 0; k < train.test_set_size; ++k)
        {
            net_fprop(net, train.test_input_set[k]);

            float* target = train.test_output_set[k];

            int target_idx = 0;
            float max_target = target[0];
            int output_idx = 0;
            float max_output = net.out_a[0];

            for(int i = 1; i < net.out_size; ++i)
            {
                if(target[i] > max_target)
                {
                    target_idx = i;
                    max_target = target[i];
                }

                if(net.out_a[i] > max_output)
                {
                    output_idx = i;
                    max_output = net.out_a[i];
                }
            }
            match_count += target_idx == output_idx;
        }
        printf("epoch %d: %d / %d [%.2f%%] matches\n", _i, match_count, train.test_set_size, 100.f * match_count / train.test_set_size);
    }
}

void dump_example(const char* prefix, float* input, float* output, int width, int height)
{
    int label = 0;
    float max_output = 0.f;

    for(int i = 0; i < 10; ++i)
    {
        if(output[i] > max_output)
        {
            max_output = output[i];
            label = i;
        }
    }
    char buf[1024];
    int n = snprintf(buf, sizeof(buf), "%s_%d.pgm", prefix, label);
    assert(n < (int)sizeof(buf) && n > 0);
    FILE* fd = fopen(buf, "wb");
    assert(fd);
    n = snprintf(buf, sizeof(buf), "P5 %d %d 255 ", width, height);
    assert(n < (int)sizeof(buf) && n > 0);
    int nw = fwrite(buf, 1, n, fd);
    assert(nw == n);
    unsigned char* image = (unsigned char*)malloc(width * height);

    for(int i = 0; i < width * height; ++i)
        image[i] = input[i] * 255.f;
    nw = fwrite(image, 1, width * height, fd);
    assert(nw == width * height);
    fclose(fd);
}

// https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit

void test_semeion()
{
    printf("semeion data set test\n");

    Net net;
    net_init(net, 16 * 16, 30, 10);

    Train train;
    train.epoch_count = 30;
    train.batch_size = 10;
    train.eta = 3.f;

    FILE* fd = fopen("semeion.data", "r");

    if(!fd)
    {
        printf("semeion.data file missing\n");
        exit(1);
    }

    int set_size = 1593;
    float** input_set = (float**)malloc(sizeof(float*) * set_size);
    float** output_set = (float**)malloc(sizeof(float*) * set_size);

    for(int i = 0; i < set_size; ++i)
    {
        float* input = (float*)malloc(sizeof(float) * 16 * 16);
        float* output = (float*)malloc(sizeof(float) * 10);

        for(int k = 0; k < 16 * 16; ++k)
        {
            float f;
            int n = fscanf(fd, "%f", &f);
            assert(n == 1);
            input[k] = f;
        }

        for(int k = 0; k < 10; ++k)
        {
            int n = fscanf(fd, "%f", output + k);
            assert(n == 1);
        }

        input_set[i] = input;
        output_set[i] = output;
    }

    fclose(fd);

    // validate if the data set was loaded correctly
    dump_example("se0", input_set[55], output_set[55], 16, 16);
    dump_example("se1", input_set[1111], output_set[1111], 16, 16);
    dump_example("se3", input_set[881], output_set[881], 16, 16);
    dump_example("se4", input_set[404], output_set[404], 16, 16);

    shuffle_set(input_set, output_set, set_size);

    // use 70% of the data set for training and the rest for testing
    train.set_size = set_size * 0.7f;
    train.test_set_size = set_size - train.set_size;

    train.input_set = input_set;
    train.output_set = output_set;

    train.test_input_set = input_set + train.set_size;
    train.test_output_set = output_set + train.set_size;

    net_train(net, train);
}

void load_mnist_set(const char* filename_images, const char* filename_labels, float**& input_set, float**& output_set, int& set_size)
{
    FILE* fd = fopen(filename_images, "rb");

    if(!fd)
    {
        printf("file missing: %s\n", filename_images);
        exit(1);
    }
    {
        unsigned int x;
        int n = fread(&x, 4, 1, fd);
        assert(n == 1);
        assert(ntohl(x) == 2051); // magic number

        n = fread(&x, 4, 1, fd);
        assert(n == 1);
        set_size = ntohl(x);

        input_set = (float**)malloc(sizeof(float*) * set_size);
        output_set = (float**)malloc(sizeof(float*) * set_size);

        n = fread(&x, 4, 1, fd);
        assert(n == 1);
        assert(ntohl(x) == 28);

        n = fread(&x, 4, 1, fd);
        assert(n == 1);
        assert(ntohl(x) == 28);
    }

    for(int i = 0; i < set_size; ++i)
    {
        float* input = (float*)malloc(sizeof(float) * 28 * 28);

        for(int k = 0; k < 28 * 28; ++k)
        {
            unsigned char x;
            int n = fread(&x, 1, 1, fd);
            assert(n == 1);
            input[k] = x / 255.f;
        }
        input_set[i] = input;
    }
    fclose(fd);
    fd = fopen(filename_labels, "rb");

    if(!fd)
    {
        printf("file missing: %s\n", filename_labels);
        exit(1);
    }
    {
        unsigned int x;
        int n = fread(&x, 4, 1, fd);
        assert(n == 1);
        assert(ntohl(x) == 2049); // magic number

        n = fread(&x, 4, 1, fd);
        assert(n == 1);
        assert((int)ntohl(x) == set_size);
    }

    for(int i = 0; i < set_size; ++i)
    {
        float* output = (float*)malloc(sizeof(float) * 10);
        memset(output, 0, sizeof(float) * 10);
        unsigned char x;
        int n = fread(&x, 1, 1, fd);
        assert(n == 1);
        assert(x < 10);
        output[x] = 1.f;
        output_set[i] = output;
    }
    fclose(fd);
}

// http://yann.lecun.com/exdb/mnist/

void test_mnist()
{
    printf("MNIST data set test\n");
    Net net;
    net_init(net, 28 * 28, 30, 10);
    Train train;
    train.epoch_count = 30;
    train.batch_size = 10;
    train.eta = 3.f;
    load_mnist_set("train-images-idx3-ubyte", "train-labels-idx1-ubyte", train.input_set, train.output_set, train.set_size);
    load_mnist_set("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", train.test_input_set, train.test_output_set, train.test_set_size);
    assert(train.set_size == 60000);
    assert(train.test_set_size == 10000);
    dump_example("mn_train0", train.input_set[2], train.output_set[2], 28, 28);
    dump_example("mn_train1", train.input_set[500], train.output_set[500], 28, 28);
    dump_example("mn_test0",  train.test_input_set[100], train.test_output_set[100], 28, 28);
    dump_example("mn_test1",  train.test_input_set[4], train.test_output_set[4], 28, 28);
    net_train(net, train);
}

int main()
{
    std::random_device rd;
    _rng.seed(rd());
    test_semeion();
    test_mnist();
    return 0;
}
