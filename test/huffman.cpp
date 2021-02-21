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
    int symbol;
    int weight;
};

struct CodeWord
{
    int symbol;
    int length;
    uint32_t value;
};

bool compare(Node& lhs, Node& rhs)
{
    return lhs.weight > rhs.weight;
}

bool compare_length(CodeWord& lhs, CodeWord& rhs)
{
    return lhs.length < rhs.length;
}

using Queue = std::priority_queue<Node, std::vector<Node>, decltype(&compare)>;

#define SIZE 256

void register_word(std::vector<CodeWord>& list, Node* node, int depth)
{
    if(node->left)
    {
        assert(node->right);
        register_word(list, node->left, depth + 1);
        register_word(list, node->right, depth + 1);
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
    std::vector<CodeWord> words;
    register_word(words, &tree, 0);
    return words;
}

int main()
{
    std::string input_str = "jfklaf hadaaaaaaabbbfdljfdfjdjfhhhff ";
    int weights[SIZE] = {};

    for(char c : input_str)
    {
        assert(c >= 0);
        weights[(int)c] += 1;
    }

    Queue queue(compare);

    for(int i = 0; i < SIZE; ++i)
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

    Node root = queue.top();
    std::vector<CodeWord> code_words = to_array(root);
    std::sort(code_words.begin(), code_words.end(), &compare_length);
    assert(code_words.back().length <= 32);
    uint32_t value = 0;
    int prev_length = code_words.front().length;

    // converting huffman tree into a canonical huffman code
    // interesting problem is to prove that this algorithm produces a prefix code

    for(CodeWord& w : code_words)
    {
        int ldiff = w.length - prev_length;

        if(ldiff)
            value = value << ldiff;

        w.value = value;
        value += 1;
        // assert for overflow
        prev_length = w.length;
    }

    for(CodeWord w : code_words)
        printf("'%c' : %d : %d\n", w.symbol, w.length, w.value);

    // now we have a huffman tree in a 'canonical' form
    // time to do bit-packing

    // layout:
    // word1 = (a3, a2, a1, b2, b1, c4, ...), word2 = (e2, e1, f3, f2, f1, ...), ...
    // a3 is the MSB of word1, e3 is the MSB of word2, ...

    std::vector<uint32_t> words;
    uint32_t word = 0;
    int word_bit_pos = 31;

    for(char c : input_str)
    {
        CodeWord code = {};
        for(CodeWord& w : code_words)
        {
            if(w.symbol == c)
            {
                code = w;
                break;
            }
        }
        assert(code.symbol == c);
        uint32_t value = code.value;
        int length = code.length;

        while(length > 0)
        {
            int num_bits = std::min(length, word_bit_pos + 1);
            length -= num_bits;
            uint32_t part = value >> length;
            uint32_t mask = (1 << length) - 1;
            value = value & mask;
            word_bit_pos -= num_bits;
            part = part << (word_bit_pos + 1);
            word = word | part;

            if(word_bit_pos < 0)
            {
                assert(word_bit_pos == -1);
                words.push_back(word);
                word_bit_pos = 31;
                word = 0;
            }
        }
    }

    if(word_bit_pos != 31)
        words.push_back(word);

    printf("input  bytes: %d\n", (int)(input_str.size()));
    printf("output bytes: %d\n", (int)(words.size() * 4));
    printf("compression ratio (no header): %f\n", (float)input_str.size() / (words.size() * 4));

    // test the encoding
    {
        // auxiliary data structure used for binary search
        // codes are already sorted by the numerical value, so no additional sorting is needed
        // one useful feature of canonical huffman code is that the codes have unique numerical values
        // (even if they differ in length)
        // this is very inefficient approach but will suffice
        std::vector<uint32_t> code_values;

        for(CodeWord& cw : code_words)
            code_values.push_back(cw.value);

        std::string decoded;

        int word_pos = 0;
        int word_bit_pos = 31;
        uint32_t value = 0;
        int value_bit_length = 0;

        while(decoded.size() != input_str.size())
        {
            assert(word_pos < (int)words.size());
            uint32_t bit = (words[word_pos] >> word_bit_pos) & 1;
            value = (value << 1) | bit;
            value_bit_length += 1;

            auto it = std::lower_bound(code_values.begin(), code_values.end(), value);

            if(it != code_values.end() && *it == value)
            {
                CodeWord& cw = code_words[it - code_values.begin()];

                if(cw.length == value_bit_length)
                {
                    value = 0;
                    value_bit_length = 0;
                    decoded.push_back(cw.symbol);
                }
            }
            word_bit_pos -= 1;

            if(word_bit_pos < 0)
            {
                assert(word_bit_pos == -1);
                word_pos += 1;
                word_bit_pos = 31;
            }
        }
        assert((word_pos == (int)words.size() && word_bit_pos == 31) || (word_pos == (int)words.size() - 1));
        printf("input   str: %s\n", input_str.c_str());
        printf("decoded str: %s\n", decoded.c_str());
    }
    return 0;
}
