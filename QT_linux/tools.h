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

typedef struct Node {
    QLabel * label;
    Node* next = NULL;
} Node;

typedef struct Node* QlabelList;

bool insert_intail(QlabelList& head,QLabel* label);
void clear(QlabelList& head);

class Qlabel_List
{
public:
    Qlabel_List();
};

#endif // TOOLS_H
