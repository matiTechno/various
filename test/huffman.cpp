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
        weights[(int)c] += 1;

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

    std::vector<uint32_t> words;
    uint32_t word = 0;
    int bits_covered = 0;

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
        int next_bit = 0;

        while(next_bit != code.length)
        {
            uint32_t part = code.value >> next_bit;
            part = part << bits_covered;
            word = word | part;
            int bits_added = std::min(code.length - next_bit, 32 - bits_covered);
            next_bit += bits_added;
            bits_covered += bits_added;

            if(bits_covered == 32)
            {
                words.push_back(word);
                bits_covered = 0;
                word = 0;
            }
        }
    }

    if(bits_covered)
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
        std::vector<uint32_t> code_values;

        for(CodeWord& cw : code_words)
            code_values.push_back(cw.value);

        std::string decoded;

        int word_pos = 0;
        int bits_covered = 0;
        uint32_t value = 0;
        int value_bit_length = 0;

        while(decoded.size() != input_str.size())
        {
            assert(word_pos < (int)words.size());
            uint32_t bit = (words[word_pos] >> bits_covered) & 1;
            value = value | (bit << value_bit_length);
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
            bits_covered += 1;

            if(bits_covered == 32)
            {
                word_pos += 1;
                bits_covered = 0;
            }
        }
        assert((word_pos == (int)words.size() && bits_covered == 0) || (word_pos == (int)words.size() - 1));
        printf("input   str: %s\n", input_str.c_str());
        printf("decoded str: %s\n", decoded.c_str());
    }
    return 0;
}
