#include <stdio.h>
#include <queue>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <string>

struct Node
{
    Node* left;
    Node* right;
    uint8_t symbol;
    int weight;
};

struct CodeWord
{
    uint8_t symbol;
    int length;
    uint32_t value;
};

bool compare_weight(Node& lhs, Node& rhs)
{
    return lhs.weight > rhs.weight;
}

using Queue = std::priority_queue<Node, std::vector<Node>, decltype(&compare_weight)>;

void register_code_word(std::vector<CodeWord>& list, Node* node, int depth)
{
    if(node->left)
    {
        assert(node->right);
        register_code_word(list, node->left, depth + 1);
        register_code_word(list, node->right, depth + 1);
        return;
    }
    assert(node->right == nullptr);
    CodeWord w;
    w.symbol = node->symbol;
    w.length = depth;
    list.push_back(w);
}

std::vector<CodeWord> to_array(Node tree)
{
    std::vector<CodeWord> code_words;
    register_code_word(code_words, &tree, 0);
    return code_words;
}

// API

struct CompressedFile
{
    int num_code_words;
    std::vector<uint8_t> symbols;
    std::vector<int> length_counts;
    std::vector<uint8_t> data;
};

CompressedFile compress(const std::vector<uint8_t>& input_data)
{
    int weights[256] = {};

    for(uint8_t c : input_data)
        weights[(int)c] += 1;

    Queue queue(compare_weight);

    for(int i = 0; i < 256; ++i)
    {
        if(weights[i] == 0)
            continue;
        Node node = {};
        node.symbol = i;
        node.weight = weights[i];
        queue.push(node);
    }

    while(queue.size() > 1)
    {
        Node lhs = queue.top();
        queue.pop();
        Node rhs = queue.top();
        queue.pop();
        Node node;
        node.left = (Node*)malloc(sizeof(Node));
        node.right = (Node*)malloc(sizeof(Node));
        *(node.left) = lhs;
        *(node.right) = rhs;
        node.weight = lhs.weight + rhs.weight;
        queue.push(node);
    }

    // convert huffman code into a canonical representation
    // interesting problem is to prove that this algorithm produces a valid prefix code

    std::vector<CodeWord> code_words = to_array(queue.top());

    std::sort(code_words.begin(), code_words.end(),
              [](CodeWord& lhs, CodeWord& rhs){return lhs.length < rhs.length;});

    printf("max code word length = %d\n", code_words.back().length);
    // todo, how likely are longer code words in a typical application?
    assert(code_words.back().length <= 32);
    uint32_t value = 0;
    int prev_length = code_words.front().length;

    // todo, assert for overflow
    for(CodeWord& w : code_words)
    {
        int ldiff = w.length - prev_length;
        value = value << ldiff;
        w.value = value;
        value += 1;
        prev_length = w.length;
    }

    //for(CodeWord &w : code_words)
        //printf("'%c' : %d : %d\n", w.symbol, w.length, w.value);

    // bit-packing
    // layout:
    // byte1 = (a3, a2, a1, b2, b1, c4, ...), byte2 = (e2, e1, f3, f2, f1, ...), ...
    // a3 is the MSB of byte1, e3 is the MSB of byte2, ...
    // (a3, a2, a1) represents a single code word 'a', where a3 is the MSB of 'a'

    std::vector<uint8_t> bytes;
    uint8_t byte = 0;
    int bit_pos = 7;

    for(uint8_t c : input_data)
    {
        uint32_t value = 0;
        int length = 0;

        for(CodeWord& w : code_words)
        {
            if(w.symbol == c)
            {
                value = w.value;
                length = w.length;
                break;
            }
        }
        assert(length);

        while(length > 0)
        {
            int num_bits = std::min(length, bit_pos + 1);
            length -= num_bits;
            uint32_t part = value >> length;
            uint32_t mask = (1 << length) - 1;
            value = value & mask;
            bit_pos -= num_bits;
            part = part << (bit_pos + 1);
            byte = byte | part;

            if(bit_pos < 0)
            {
                bytes.push_back(byte);
                bit_pos = 7;
                byte = 0;
            }
        }
    }
    if(bit_pos != 7)
        bytes.push_back(byte);

    CompressedFile file;
    file.num_code_words = input_data.size();
    file.data = bytes;
    file.length_counts.resize(code_words.back().length, 0);

    for(CodeWord w : code_words)
    {
        file.symbols.push_back(w.symbol);
        file.length_counts[w.length - 1] += 1;
    }
    return file;
}

std::vector<uint8_t> decompress(const CompressedFile& file)
{
    // create a list of code words

    std::vector<CodeWord> code_words;
    {
        std::size_t symbol_pos = 0;
        uint32_t value = 0;
        int prev_length = 0;
        int length = 0;

        for(int count : file.length_counts)
        {
            length += 1;

            if(count == 0)
                continue;

            if(value)
                value = value << (length - prev_length);

            prev_length = length;

            while(count > 0)
            {
                count -= 1;
                CodeWord w;
                w.symbol = file.symbols[symbol_pos];
                w.length = length;
                w.value = value;
                code_words.push_back(w);
                symbol_pos += 1;
                value += 1;
            }
        }
        assert(symbol_pos == file.symbols.size());
    }

    // auxiliary data structure used for binary search
    // (very inefficient decoding algorithm but will suffice)

    std::vector<uint32_t> cw_values;

   for(CodeWord& w : code_words)
        cw_values.push_back(w.value);

    // Code words are already sorted by the numerical value, so no additional sorting is needed.
    // One useful feature of canonical huffman code is that code words have unique numerical values
    // (even if they differ in length).

    std::vector<uint8_t> output;
    std::size_t byte_pos = 0;
    int bit_pos = 7;
    uint32_t value = 0;
    int length = 0;

    while(output.size() != (std::size_t)file.num_code_words)
    {
        assert(byte_pos < file.data.size());
        uint32_t bit = (file.data[byte_pos] >> bit_pos) & 1;
        value = (value << 1) | bit;
        length += 1;

        auto it = std::lower_bound(cw_values.begin(), cw_values.end(), value);

        if(it != cw_values.end() && *it == value)
        {
            CodeWord& w = code_words[it - cw_values.begin()];

            if(w.length == length)
            {
                value = 0;
                length = 0;
                output.push_back(w.symbol);
            }
        }
        bit_pos -= 1;

        if(bit_pos < 0)
        {
            byte_pos += 1;
            bit_pos = 7;
        }
    }
    return output;
}

std::vector<uint8_t> file_to_bytes(const CompressedFile& file)
{
    int symbol_count = file.symbols.size();
    std::vector<uint8_t> bytes;
    // bs - byte size
    int bs1 = sizeof(int); // number of code words
    int bs2 = sizeof(int); // number of symbols
    int bs3 = file.symbols.size();
    int bs4 = file.length_counts.size() * sizeof(int);
    int bs5 = file.data.size();
    bytes.resize(bs1 + bs2 + bs3 + bs4 + bs5);
    uint8_t* dst = bytes.data();
    memcpy(dst, &file.num_code_words, bs1);
    dst += bs1;
    memcpy(dst, &symbol_count, bs2);
    dst += bs2;
    memcpy(dst, file.symbols.data(), bs3);
    dst += bs3;
    memcpy(dst, file.length_counts.data(), bs4);
    dst += bs4;
    memcpy(dst, file.data.data(), bs5);
    dst += bs5;
    assert(dst = bytes.data() + bytes.size());
    return bytes;
}

CompressedFile file_from_bytes(const std::vector<uint8_t>& bytes)
{
    const uint8_t* src = bytes.data();
    CompressedFile file;
    int bs1 = sizeof(int);
    int bs2 = sizeof(int);

    memcpy(&file.num_code_words, src, bs1);
    src += bs1;

    int symbol_count = 0;
    memcpy(&symbol_count, src, bs2);
    src += bs2;

    file.symbols.resize(symbol_count);
    memcpy(file.symbols.data(), src, symbol_count);
    src += symbol_count;

    while(symbol_count)
    {
        int count = 0;
        memcpy(&count, src, sizeof(int));
        src += sizeof(int);
        symbol_count -= count;
        file.length_counts.push_back(count);
    }

    assert(symbol_count == 0);
    int data_size = &bytes.back() - src + 1;
    file.data.resize(data_size);
    memcpy(file.data.data(), src, data_size);
    return file;
}

std::vector<uint8_t> read_fs_file(const char* filename)
{
    FILE* file = fopen(filename, "rb");

    if(!file)
    {
        printf("couldn't open %s for read\n", filename);
        exit(1);
    }
    int rc = fseek(file, 0, SEEK_END);
    assert(rc == 0);
    int size = ftell(file);

    if(size == 0)
    {
        printf("empty file %s, exiting\n", filename);
        exit(1);
    }

    rewind(file);
    std::vector<uint8_t> data(size);
    fread(data.data(), 1, size, file);
    fclose(file);
    return data;
}

void write_fs_file(const char* filename, std::vector<uint8_t>& data)
{
    FILE* file = fopen(filename, "wb");

    if(!file)
    {
        printf("couldn't open %s for write\n", filename);
        exit(1);
    }
    std::size_t size = fwrite(data.data(), 1, data.size(), file);

    if(size != data.size())
    {
        printf("fwrite() error on %s\n", filename);
        exit(1);
    }
    fclose(file);
}

void arg_error()
{
    printf("expecting 0 or 3 arguments, example usage:\n");
    printf("a.out compress file.txt file.bin\n");
    printf("a.out decompress file.bin file.txt\n");
}

int main(int argc, char** argv)
{
    if(argc != 1 && argc != 4)
    {
        arg_error();
        return 1;
    }

    if(argc == 4)
    {
        if(strcmp("compress", argv[1]) == 0)
        {
            std::vector<uint8_t> input = read_fs_file(argv[2]);
            CompressedFile file = compress(input);
            std::vector<uint8_t> output = file_to_bytes(file);
            printf("compression ratio: %f\n", (float)input.size() / output.size());
            write_fs_file(argv[3], output);
            return 0;
        }
        else if(strcmp("decompress", argv[1]) == 0)
        {
            std::vector<uint8_t> input = read_fs_file(argv[2]);
            CompressedFile file = file_from_bytes(input);
            std::vector<uint8_t> output = decompress(file);
            printf("compression ratio: %f\n", (float)output.size() / input.size());
            write_fs_file(argv[3], output);
            return 0;
        }
        else
        {
            arg_error();
            return 1;
        }
    }

    // otherwise do a sample test run...

    const char* str = R"(
        The calculus of variations is a field of mathematical analysis that uses variations, which are
        small changes in functions and functionals, to find maxima and minima of functionals: mappings
        from a set of functions to the real numbers.[a] Functionals are often expressed as definite
        integrals involving functions and their derivatives. Functions that maximize or minimize
        functionals may be found using the Euler–Lagrange equation of the calculus of variations.
        A simple example of such a problem is to find the curve of shortest length connecting two
        points. If there are no constraints, the solution is a straight line between the points.
        However, if the curve is constrained to lie on a surface in space, then the solution is less
        obvious, and possibly many solutions may exist. Such solutions are known as geodesics. A
        related problem is posed by Fermat's principle: light follows the path of shortest optical
        length connecting two points, where the optical length depends upon the material of the
        medium. One corresponding concept in mechanics is the principle of least/stationary action.
        Many important problems involve functions of several variables. Solutions of boundary value
        problems for the Laplace equation satisfy the Dirichlet principle. Plateau's problem requires
        finding a surface of minimal area that spans a given contour in space: a solution can often
        be found by dipping a frame in a solution of soap suds. Although such experiments are
        relatively easy to perform, their mathematical interpretation is far from simple: there may
        be more than one locally minimizing surface, and they may have non-trivial topology.
        )";

    std::vector<uint8_t> input(strlen(str));
    memcpy(input.data(), str, strlen(str));
    CompressedFile file = compress(input);
    std::vector<uint8_t> bytes = file_to_bytes(file);
    printf("compression ratio = %f\n", (float)input.size() / bytes.size());
    CompressedFile recovered_file = file_from_bytes(bytes);
    std::vector<uint8_t> output = decompress(recovered_file);
    assert(input == output);
    return 0;
}
