#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <queue>
#include <functional>
#include <unordered_map>
#include <set>
#include <utility>
#include <limits>
#include <sstream>

/* This code containts the required functions for the first four levels 
 * of the challenge. I address the fifth question in a seperate Python code.
 * Note this is only written to meet the requirements of the problem, and 
 * is not polished enough to be suitable for production.
 * Currently, I use global variables that I would not use unless I had to, 
 * in a production code. So, if I had more time, I would make this code
 * Object Oriented and would remove the global variables. *
 */
using namespace std;

// global variable, showing the length of 
// the first dimension of the chess board. Board is nxn
static int n = 8;

// Struct Coordinate encodes each square in the board
struct Coordinate {
    int x, y;
    Coordinate(int a, int b) : x{a}, y{b} {}
    Coordinate() : x{0}, y{0} {}
    bool operator==(const Coordinate& that) const {
        return (x == that.x && y == that.y);
    }
    bool operator!=(const Coordinate& that) const {
        return !(*this == that);
    }
    string str() const {
        return ("x: " + to_string(x) + ", y: " + to_string(y));
    } 
};

// Only for the purpose of printing the board for level_1 question
struct BoardPrinter {
    vector<vector<char>> layout;
    BoardPrinter(Coordinate& c) : layout{vector<vector<char>>(n, vector<char>(n, '.'))} {
        layout[c.x][c.y] = 'K';
    }
    void move(Coordinate& c1, Coordinate& c2) {
        layout[c1.x][c1.y] = '.';
        layout[c2.x][c2.y] = 'K';
        print();
    }
    string print() {
        stringstream ss;
        for (int i = 0; i < n; ++i) {
           for (int j = 0; j < n - 1; ++j)
               ss << layout[i][j] << ' ';
           ss << layout[i][n - 1] << '\n';
        }
        return ss.str();
    }          
};

// Checks whether a given coordinate is within the board borders
bool valid_pos(const Coordinate& c) {
    return (c.x >= 0 && c.y >= 0 && c.x < n && c.y < n);
}

/* Produces a list of squares that the Knight at coordinate c 
* can transition to, in one knight move.
* For a plain board (without obstacles and marks) the last two arguments
* are null. Otherwise, if c is teleport, the other teleport nodes will also 
* be added to the list of its neighbours.
* A better choice would be for it to return an unordered_set to remove the 
* duplicates, and the node itself, as a result of teleporting;
* But this has been taken care of in Dijekstra algorithm. So, has no harm.
*/ 
vector<Coordinate> children(const Coordinate& c, 
                            function<bool(const Coordinate&)> is_teleport = nullptr,
                            vector<Coordinate>* teleport_coordinates = nullptr) {
    vector<Coordinate> neighbours;
    const vector<vector<int>> shift = {
        {1, 2}, {-1, 2}, {1, -2}, {-1, -2},
        {2, 1}, {2, -1}, {-2, 1}, {-2, -1}};
    for (auto& s : shift) {
        Coordinate t(c.x + s[0], c.y + s[1]);
        if (valid_pos(t))
            neighbours.emplace_back(t);
    }
    if (is_teleport && is_teleport(c) && teleport_coordinates) {
        neighbours.insert(neighbours.end(), teleport_coordinates->begin(), 
                          teleport_coordinates->end());
    }
    return neighbours;
}

/* validates the correctness of the length of a move
* assuming a valid src and dst, on a plain board. 
* Note again that this function does not check for 
* the validity of its arguments.
*/
bool valid_move(const Coordinate& s, const Coordinate& t) {
    int delta_x = t.x - s.x;
    int delta_y = t.y - s.y;
    return ((abs(delta_x) == 2 && abs(delta_y) == 1) ||
            (abs(delta_x) == 1 && abs(delta_y) == 2)); 
}

/* Checks if a given path is correct (level_1 question),
* and prints the state of the board taking this path.
* It also includes checking for the validity of each of
* the squares along the path. This check could have 
* perfectly moved to the valid_move function, but then every 
* square would have been checked two times that did not make me happy. 
*/
bool valid_path(vector<Coordinate>& path) {
    if (!valid_pos(path[0])) {
        cout << "invalid source.\n";
        return false;
    }
    BoardPrinter layout(path[0]);
    cout << layout.print() << '\n';
    for (int i = 0; i < path.size() - 1; ++i) {
        if (!valid_move(path[i], path[i + 1])) {
          cout << "invalid move from " << path[i].str()
               << " to " << path[i + 1].str() << ".\n";
          return false;
        }
        layout.move(path[i], path[i + 1]);
        cout << layout.print() << '\n';
    }
    return true;
}

/* Runs BFS to find a path between two given squares.
* I chose BFS which gives us the shortest path, to be able to 
* use it later for testing the correctness of Dijekstra for level 4.
* It returns the length of the shortest path; and reports the path
* in its last argument(path).
*/
int bfs(Coordinate& src, Coordinate& dst, vector<Coordinate>& path) {
    if (!valid_pos(src) || !valid_pos(dst)) {
        cout << "Wrong source or destination. \n";
        return 0;
    }
    queue<Coordinate> candidates;
    vector<vector<Coordinate>> parent(n, vector<Coordinate>(n));
    vector<vector<bool>> visited(n, vector<bool>(n, false));
    candidates.push(src);
    visited[src.x][src.y] = true;
    while (!candidates.empty() && !visited[dst.x][dst.y]) {
        auto cur = candidates.front();
        candidates.pop();
        for (auto& child : children(cur))
            if (!visited[child.x][child.y]) {
                visited[child.x][child.y] = true;
                parent[child.x][child.y] = cur;
                candidates.emplace(child);
                if (child == dst)
                    break;
            }
    }
    if (visited[dst.x][dst.y]) {
        for (auto c = dst; c != src; c = parent[c.x][c.y])
            path.push_back(c);
        path.push_back(src);
        reverse(path.begin(), path.end());
    }
    return visited[dst.x][dst.y] ? path.size() - 1 : numeric_limits<int>::max();
}

// Provides a rather naive hash function for the Coordinate
// type to be later used in a map.
struct CoordHash {
    int operator()(const Coordinate& c) const {
        return (c.x * n + c.y);
    }
};

typedef pair<Coordinate, int> CoorWeightType;
// Less function for the map to be able to lookup the square with 
// the minimum weight calculated so far, in Dijekstra algorithm.
struct WeightCompare {
    bool operator()(const CoorWeightType& lhs, const CoorWeightType& rhs) {
        return lhs.second < rhs.second;
    }
};

/* I chose Dijekstra to find the shortest path in level 4,
 * because the way I encoded the weights, there is no negative edge
 * and therefore no negative cycle; and that we know Dijekstra is 
 * easy to implement.
 * The last two arguments are used for the special case of 
 * teleport nodes in children function.
 * weight argument (4th) finds the weight of an edge.
 * the function returns the length of the shortest path and fills in 
 * the third argument with the path that it found.
 * An important issue here is that the map that stores squares and the
 * shortest path to them (found so far) does not provide a constant time
 * min function and I have to go over the entire map each time to find the 
 * node with the shortest path. If I had more time, I would design 
 * and indexed min heap for this purpose (also exists in Boost I think).
 */
int dijekstra(const Coordinate& src, const Coordinate& dst, vector<Coordinate>& path,
              function<int(const Coordinate&, const Coordinate&)> weight, 
              function<bool(const Coordinate&)> is_teleport = nullptr,
              vector<Coordinate>* teleport_coordinates = nullptr) {
    if (!valid_pos(src) || !valid_pos(dst)) {
        cout << "Wrong source or destination. \n";
        return numeric_limits<int>::max();
    }
    // Stores the coordinates along with their minimum distance 
    //from the src, found so far.
    unordered_map<Coordinate, int, CoordHash> coor_weight_map;
    // For each square, stores the parent square that provides the 
    // shortest distance from the source.
    unordered_map<Coordinate, Coordinate, CoordHash> path_to;
    coor_weight_map[src] = 0;
    // Stores the connected squares to the source, found so far.
    vector<vector<bool>> visited(n, vector<bool>(n, false));
    while (!coor_weight_map.empty()) {
        auto cur = min_element(coor_weight_map.begin(), 
                               coor_weight_map.end(), WeightCompare());
        visited[(cur->first).x][(cur->first).y] = true;
        if (cur->first == dst)
            break;
        // relax the current node
        auto proxy_weight = cur->second;
        auto neighbours = children(cur->first, 
                                   is_teleport, teleport_coordinates);
        for (auto& child : neighbours) {
            long long edge_weight = weight(cur->first, child);
            if (edge_weight < numeric_limits<int>::max() &&
                (!visited[child.x][child.y] ||
                ((coor_weight_map.find(child) != coor_weight_map.end() &&
                edge_weight + proxy_weight < coor_weight_map[child])))) {
                coor_weight_map[child] = edge_weight + proxy_weight;
                path_to[child] = cur->first;
            }
        }
        coor_weight_map.erase(cur);
    }
    if (path_to.find(dst) != path_to.end()) {
        for (auto coord = dst; coord != src; coord = path_to[coord]) {
            path.push_back(coord);
        }
        path.push_back(src);
        reverse(path.begin(), path.end());
    }
    return (path_to.find(dst) != path_to.end()) ? 
           coor_weight_map[dst] : numeric_limits<int>::max();
}


/*Checks the validity of an edge given the rules of a marked board
 * (level_4). It looks for B's on the knight move.
 */
bool edge_feasible(const Coordinate& s, const Coordinate& t,
                   function<bool(int, int)> barrier) {
    if(!valid_move(s, t))
        return false;
    bool path_1 = barrier(s.x, t.y) || barrier(t.x, t.y);
    bool path_2 = barrier(t.x, s.y) || barrier(t.x, t.y);
    int delta_x = t.x - s.x;
    if (abs(delta_x) == 2) {
        path_1 = path_1 || barrier(int((s.x + t.x) / 2), t.y);
        path_2 = path_2 || barrier(int((s.x + t.x) / 2), s.y);
    }
    else { //abs(delta_x) == 1 
        path_1 = path_1 || barrier(s.x, int((s.y + t.y) / 2));
        path_2 = path_2 || barrier(t.x, int((s.y + t.y) / 2));
    }
    return !(path_1 && path_2);
}

/* Calculates the cost of an edge, given a marked board.
 * Specifically, it lookes for T's, B's, R's, W's, and L's.
 * By default, the cost is 1.
 * Note, this returns another function that does not require the 
 * board marks, and is sent to Dijekstra.
 * This is so that Dijekstra is compatible for both marked (level_4) and 
 * plain (level_3) boards. It can potentially benefit from a better design/API.
 */
function<int(const Coordinate&, const Coordinate&)> cost_evaluator(vector<vector<char>>& rules_matrix ) {
    return [&rules_matrix] (const Coordinate& s, const Coordinate& t) -> int {
        auto rule_t = rules_matrix[t.x][t.y];
        auto rule_s = rules_matrix[s.x][s.y];
        if (rule_t == 'T' && rule_s == 'T')
            return 0;
        bool good_edge = edge_feasible(s, t,
                [&rules_matrix](int x, int y) {
                    return rules_matrix[x][y] == 'B';});
        if (rule_t == 'B' || rule_t == 'R' || !good_edge)
            return numeric_limits<int>::max();
        int cost = 1;
        if (rule_t == 'W')
           cost = 2;
        else if (rule_t == 'L')
           cost = 5;
        return cost; 
    };
}

// Given an example like the one provided in level_4, returns the board layout.
// Also calculates a list of the telport squares.
vector<vector<char>> read_board(vector<string>& board_marks_str,
                                vector<Coordinate>& teleport_squares) {
    vector<vector<char>> board_marks(board_marks_str.size(),
            vector<char>((1 + board_marks_str[0].size() / 2)));
    for (int i = 0; i < board_marks_str.size(); ++i)
        for (int j = 0; j < board_marks_str[i].size(); j += 2) {
            board_marks[i][j / 2] = board_marks_str[i][j];
            if (board_marks_str[i][j] == 'T')
                teleport_squares.emplace_back(Coordinate(i, j / 2));
        }
    cout << "Teleport marks are at: \n";
    for (auto& c : teleport_squares)
        cout << c.str() << '\n';
    return board_marks;
}

// The board example of level_4
vector<string> board_marks_str = {
    ". . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .",
    ". . . . . . . . B . . . L L L . . . . . . . . . . . . . . . . .",
    ". . . . . . . . B . . . L L L . . . L L L . . . . . . . . . . .",
    ". . . . . . . . B . . . L L L . . L L L . . . R R . . . . . . .",
    ". . . . . . . . B . . . L L L L L L L L . . . R R . . . . . . .",
    ". . . . . . . . B . . . L L L L L L . . . . . . . . . . . . . .",
    ". . . . . . . . B . . . . . . . . . . . . R R . . . . . . . . .",
    ". . . . . . . . B B . . . . . . . . . . . R R . . . . . . . . .",
    ". . . . . . . . W B B . . . . . . . . . . . . . . . . . . . . .",
    ". . . R R . . . W W B B B B B B B B B B . . . . . . . . . . . .",
    ". . . R R . . . W W . . . . . . . . . B . . . . . . . . . . . .",
    ". . . . . . . . W W . . . . . . . . . B . . . . . . T . . . . .",
    ". . . W W W W W W W . . . . . . . . . B . . . . . . . . . . . .",
    ". . . W W W W W W W . . . . . . . . . B . . R R . . . . . . . .",
    ". . . W W . . . . . . . . . . B B B B B . . R R . W W W W W W W",
    ". . . W W . . . . . . . . . . B . . . . . . . . . W . . . . . .",
    "W W W W . . . . . . . . . . . B . . . W W W W W W W . . . . . .",
    ". . . W W W W W W W . . . . . B . . . . . . . . . . . . B B B B",
    ". . . W W W W W W W . . . . . B B B . . . . . . . . . . B . . .",
    ". . . W W W W W W W . . . . . . . B W W W W W W B B B B B . . .",
    ". . . W W W W W W W . . . . . . . B W W W W W W B . . . . . . .",
    ". . . . . . . . . . . B B B . . . . . . . . . . B B . . . . . .",
    ". . . . . R R . . . . B . . . . . . . . . . . . . B . . . . . .",
    ". . . . . R R . . . . B . . . . . . . . . . . . . B . T . . . .",
    ". . . . . . . . . . . B . . . . . R R . . . . . . B . . . . . .",
    ". . . . . . . . . . . B . . . . . R R . . . . . . . . . . . . .",
    ". . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .",
    ". . . . . . . . . . . B . . . . . . . . . . R R . . . . . . . .",
    ". . . . . . . . W W . . . . . . . . . B . . . . . . T . . . . .",
    ". . . W W W W W W W . . . . . . . . . B . . . . . . . . . . . .",
    ". . . W W W W W W W . . . . . . . . . B . . R R . . . . . . . .",
    ". . . W W . . . . . . . . . . B B B B B . . R R . W W W W W W W"};

