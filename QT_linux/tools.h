#ifndef TOOLS_H
#define TOOLS_H
#include <QLabel>
#include <string>
using namespace std;

struct Struct_XY{
    string label;
    double x;
    double y;
};

struct Node {
    QLabel * qlabel;
    Node* next = NULL;
};

typedef struct Node* QlabelList;

void insert(QlabelList& head,QLabel* qlabel);
void clean(QlabelList& head);
void hideqlabels(QlabelList& head);
void showqlabels(QlabelList& head);

#endif // TOOLS_H
