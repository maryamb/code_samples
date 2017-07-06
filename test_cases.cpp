#include "shortest_path.cpp"
#include <cassert>

void valid_pos_test() {
    n = 8;
    assert(valid_pos(Coordinate(0, 0)));
    assert(valid_pos(Coordinate(0, 7)));
    assert(valid_pos(Coordinate(7, 0)));
    assert(valid_pos(Coordinate(1, 1)));
    assert(valid_pos(Coordinate(7, 7)));
    assert(!valid_pos(Coordinate(-1, 0)));
    assert(!valid_pos(Coordinate(8, 0)));
    assert(!valid_pos(Coordinate(0, 8)));
    assert(!valid_pos(Coordinate(8, 8)));
    cout << "Passed valid_pos test successfully!\n";
}

void children_test() {
    n = 32;
    auto teleport_marks = vector<Coordinate>{Coordinate(11, 26),
           Coordinate(23, 27), Coordinate(28, 26)};
    auto v1 = children(Coordinate(0, 0));
    auto v2 = vector<Coordinate>{{1, 2}, {2, 1}};
    assert(equal(v1.begin(), v1.end(), v2.begin()));

    v1 = children(Coordinate(31, 31));
    v2 = vector<Coordinate>{{30, 29}, {29, 30}};
    assert(equal(v1.begin(), v1.end(), v2.begin()));

    v1 = children(Coordinate(23, 27));
    v2 = vector<Coordinate>{{24, 29}, {22, 29},
            {24, 25}, {22, 25}, {25, 28}, {25, 26}, 
            {21, 28}, {21, 26}};
    assert(equal(v1.begin(), v1.end(), v2.begin()));

    v1 = children(Coordinate(23, 27), 
            [](const Coordinate&) {return true;}, &teleport_marks) ;
    v2.insert(v2.end(), teleport_marks.begin(), teleport_marks.end());
    assert(equal(v1.begin(), v1.end(), v2.begin()));

    cout << "Passed children test successfully!\n";
}

void valid_move_test() {
    n = 32;
    auto c = Coordinate(1, 2);
    assert(valid_move(c, Coordinate(2, 4)));
    assert(!valid_move(c, Coordinate(0, 1)));
    cout << "Passed valid_move test successfully!\n";
}

// Problem level_1
void valid_path_test() {
    n = 8;
    vector<Coordinate> path{{1, 2}, {2, 4}, {3, 6}, {2, 4}, 
            {3, 6}, {1, 7}};
    assert(valid_path(path));
    path.push_back({1, 8});
    assert(!valid_path(path));
    cout << "Passed valid_path test successfully!\n";
}

//Level_2, and level_3
void bfs_test() {
    n = 8;
    auto s = Coordinate(7, 0), t = Coordinate(7, 7);
    vector<Coordinate> short_path_bfs;
    auto bfs_len = bfs(s, t, short_path_bfs);
    cout << "BFS:\n";
    for (auto c : short_path_bfs)
        cout << c.str() << '\n';
    cout << "Dijekstra: \n";
    vector<Coordinate> short_path_dijekstra;
    auto dijekstra_len = dijekstra(s, t, short_path_dijekstra,
            [](const Coordinate&, const Coordinate&) -> int {return 1;});
    for (auto c : short_path_dijekstra) 
        cout << c.str() << '\n';
    assert(bfs_len == dijekstra_len);
    assert(short_path_bfs.back() == short_path_dijekstra.back());
    cout << "Passed bfs test successfully!\n";
}

void cost_evaluator_test() {
    n = 32;
    Coordinate c1(10, 18);
    Coordinate c2(8, 19);
    auto no_barrier = [](int, int) {return false;};
    assert(edge_feasible(c1, c2, no_barrier));
    vector<Coordinate> teleports;
    auto board_marks = read_board(board_marks_str, teleports);
    assert(board_marks.size() == n);
    assert(board_marks[0].size() == n);
    auto weight = cost_evaluator(board_marks);
    assert(weight(c1, c2) == numeric_limits<int>::max());
    c1 = {10 , 3};
    c2 = {8, 4};
    auto barrier = [&board_marks] (int x, int y) {
            return board_marks[x][y] == 'B';};
    assert(board_marks[c1.x][c1.y] == 'R');
    assert(board_marks[c2.x][c2.y] == '.');
    assert(edge_feasible(c1, c2, barrier));
    assert(edge_feasible(c2, c1, barrier));
    assert(weight(c1, c2) == 1);
    assert(weight(c2, c1) == numeric_limits<int>::max());
    c1 = {12, 3};
    c2 = {13, 5};
    assert(board_marks[c1.x][c1.y] == 'W');
    assert(board_marks[c2.x][c2.y] == 'W');
    assert(weight(c1, c2) == 2);
    c1 = {0, 11};
    c2 = {2, 12};
    assert(board_marks[c1.x][c1.y] == '.');
    assert(board_marks[c2.x][c2.y] == 'L');
    assert(weight(c1, c2) == 5);
    c1 = {11, 26};
    c2 = {23, 27};
    assert(board_marks[c1.x][c1.y] == 'T');
    assert(board_marks[c2.x][c2.y] == 'T');
    assert(weight(c1, c2) == 0);
    cout << "Passed edge_feasible , and cost_evaluator tests successfully!\n";
}

//Level_4 Problem
void dijekstra_test() {
    n = 32;
    vector<Coordinate> teleport_marks;
    auto board_marks = read_board(board_marks_str, teleport_marks);
    auto s = Coordinate(11, 26), t = Coordinate(20, 21);
    cout << "Dijekstra with board marks: \n";
    vector<Coordinate> short_path_dijekstra;
    auto dijekstra_len = dijekstra(s, t, short_path_dijekstra,
            cost_evaluator(board_marks), 
            [&board_marks](const Coordinate& c) {
                return board_marks[c.x][c.y] == 'T';},
            &teleport_marks);
    for (auto c : short_path_dijekstra)
        cout << c.str() << '\n';
    // s, and t are both teleport here:
    t = {23, 27};
    short_path_dijekstra.clear();
    dijekstra_len = dijekstra(s, t, short_path_dijekstra,
            cost_evaluator(board_marks), 
            [&board_marks](const Coordinate& c) {
                return board_marks[c.x][c.y] == 'T';},
            &teleport_marks);
    assert(dijekstra_len == 0);
    cout << "Passed Dijekstra test successfully!\n";
}
int main () {
    valid_pos_test();
    children_test();
    valid_move_test();
    valid_path_test();
    bfs_test();
    cost_evaluator_test();
    dijekstra_test();
}
