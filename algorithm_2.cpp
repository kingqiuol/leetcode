#include<iostream>
#include<vector>
#include<tr1/unordered_map>
#include<map>
#include<algorithm>
#include<math.h>
#include<stack>
#include<queue>
#include<set>
#include<climits>
#include<numeric>
#include<time.h>
#include<list>

using namespace std;
using namespace std::tr1;

/*
693. 交替位二进制数

给定一个正整数，检查他是否为交替位二进制数：换句话说，就是他的二进制数相邻的两个位数永不相等。

示例 1:
输入: 5
输出: True
解释:5的二进制数是: 101
*/
bool hasAlternatingBits(int n) {
	int m=n>>1;

	int i=0;
	int l=n;
	int max=1;
	while(l != 0){
		l=l>>1;
		max=max<<1;
		++i;
	}
	return (n^m)==(max-1);
}

/*
696. 计数二进制子串

给定一个字符串 s，计算具有相同数量0和1的非空(连续)子字符串的数量，
并且这些子字符串中的所有0和所有1都是组合在一起的。
重复出现的子串要计算它们出现的次数。

示例 1 :
输入: "00110011"
输出: 6
解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。
请注意，一些重复出现的子串要计算它们出现的次数。
另外，“00110011”不是有效的子串，因为所有的0（和1）没有组合在一起。
*/
int countBinarySubstrings(string s) {
       int* ar = new int[50000];

        int count = 1;

        int i = 0,j;

        for (j = 1;j < (int)s.size();j++)

        {

            if (s[j] == s[j-1])

                count++;

            else {

                ar[i++] = count;

                count = 1;

            }

        }

        ar[i++] = count;

        int sum = 0;

        while (--i)

        {

            sum += min(ar[i],ar[i-1]);

        }

        delete []ar;

        return sum;
}

/*
697. 数组的度

给定一个非空且只包含非负数的整数数组 nums, 数组的度的定义是指数组里任一元素出现频数的最大值。
你的任务是找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。

示例 1:
输入: [1, 2, 2, 3, 1]
输出: 2
解释: 
输入数组的度是2，因为元素1和2的出现频数最大，均为2.
连续子数组里面拥有相同度的有如下所示:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
最短连续子数组[2, 2]的长度为2，所以返回2.
*/
int findShortestSubArray(vector<int>& nums) {
	map<int,std::vector<int>> tmp;
	for(size_t i=0;i<nums.size();++i){
		tmp[nums[i]].push_back(i);
	}

	int n=0;
	int res=INT_MAX;
	for(auto it=tmp.begin(); it!=tmp.end();++it){
		int m=(int)it->second.size();
		if(n<m){
			n=m;
			res=it->second[m-1]-it->second[0]+1;
		}
		if(m==n){
			res=min(res,it->second[m-1]-it->second[0]+1);
		}
	}

	return res;
}

/*
700. 二叉搜索树中的搜索

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 
返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

例如，
给定二叉搜索树:
        4
       / \
      2   7
     / \
    1   3
和值: 2
你应该返回如下子树:
      2     
     / \   
    1   3
在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
*/
struct TreeNode {
	int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
TreeNode* searchBST(TreeNode* root, int val) {
	if(root==NULL){
		return NULL;
	}else{

		if(root->val==val){
			return root;
		}else if(root->val>val){
			return searchBST(root->left,val);
		}else{
			return searchBST(root->left,val);
		}
	}
}

/*
703. 数据流中的第K大元素

设计一个找到数据流中第K大元素的类（class）。注意是排序后的第K大元素，不是第K个不同的元素。
你的 KthLargest 类需要一个同时接收整数 k 和整数数组nums 的构造器，它包含数据流中的初始元素。
每次调用 KthLargest.add，返回当前数据流中第K大的元素。

示例:
int k = 3;
int[] arr = [4,5,8,2];
KthLargest kthLargest = new KthLargest(3, arr);
kthLargest.add(3);   // returns 4
kthLargest.add(5);   // returns 5
kthLargest.add(10);  // returns 5
kthLargest.add(9);   // returns 8
kthLargest.add(4);   // returns 8。
*/
class KthLargest {
public:
	map<int, int> signs;
	int mapSize = 0;
	int k = 0;
	KthLargest(int k, vector<int>& nums) {
		this->k = k;
		for (int i = 0; i < (int)nums.size(); i++) {
			add(nums[i]);
		}
	}

	int add(int val) {
		if (mapSize < k) {
			signs[val]++;
			mapSize++;
		}
		else if(val > signs.begin()->first) {
			int count = signs.begin()->second;
			if (count == 1) {
				signs.erase(signs.begin());
			}
			else {
				signs.begin()->second = count - 1;
			}
			signs[val]++;
		}
		return signs.begin()->first;
	}
};

/*
704. 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

示例 1:
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4
*/
int search(vector<int>& nums, int target) {
	int start=0,end=nums.size()-1;
	while(start <= end){
		int mid=(start+end)/2;
		if(nums[mid]== target){
			return mid;
		}else if(nums[mid]<target){
			start=mid+1;
		}else{
			end=mid-1;
		}
	}

	return -1;
}

/*
705. 设计哈希集合

不使用任何内建的哈希表库设计一个哈希集合
具体地说，你的设计应该包含以下的功能
	add(value)：向哈希集合中插入一个值。
	contains(value) ：返回哈希集合中是否存在这个值。
	remove(value)：将给定值从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

示例:
MyHashSet hashSet = new MyHashSet();
hashSet.add(1);         
hashSet.add(2);         
hashSet.contains(1);    // 返回 true
hashSet.contains(3);    // 返回 false (未找到)
hashSet.add(2);          
hashSet.contains(2);    // 返回 true
hashSet.remove(2);          
hashSet.contains(2);    // 返回  false (已经被删除)
*/
class MyHashSet {
public:
    /** Initialize your data structure here. */
    int N=1001;
    MyHashSet() {
        hashtable = vector<list<int>>(N);
    }
    
    void add(int key) { 
        int mod = key% N;
        if(contains(key))
            return;
        hashtable[mod].push_front(key);
    }
    
    void remove(int key) { 
        int mod = key% N;
        if(!contains(key))return;
        hashtable[mod].remove(key);
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        int mod = key% N;
        if( find(hashtable[mod].begin(),hashtable[mod].end(),key ) != hashtable[mod].end()  )
            return true;
        return false;
    }

private:
    vector<list<int>> hashtable; 
};

/*
706. 设计哈希映射

不使用任何内建的哈希表库设计一个哈希映射
具体地说，你的设计应该包含以下的功能
	put(key, value)：向哈希映射中插入(键,值)的数值对。如果键对应的值已经存在，更新这个值。
	get(key)：返回给定的键所对应的值，如果映射中不包含这个键，返回-1。
	remove(key)：如果映射中存在这个键，删除这个数值对。

示例：
MyHashMap hashMap = new MyHashMap();
hashMap.put(1, 1);          
hashMap.put(2, 2);         
hashMap.get(1);            // 返回 1
hashMap.get(3);            // 返回 -1 (未找到)
hashMap.put(2, 1);         // 更新已有的值
hashMap.get(2);            // 返回 1 
hashMap.remove(2);         // 删除键为2的数据
hashMap.get(2);            // 返回 -1 (未找到) 
*/
class MyHashMap {
private:
    struct node {
        int value;
        int key;
        node* next;
        node(int k, int val): key(k), value(val), next(NULL) {};
    };
    vector<node*> dict;
    int size = 10000;
public:
    /** Initialize your data structure here. */
    MyHashMap() {
        dict = vector<node*>(size, new node(-1, -1));
    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        int index = key % size;
        node* temp = dict[index];
        node* last_node;
        while(temp != NULL) {
            if (temp->key == key) {
                temp->value = value;
                return;
            }
            last_node = temp;
            temp = temp->next;
        }
        node* cur = new node(key, value);
        last_node->next = cur;
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int index = key % size;
        node* temp = dict[index];
        while(temp != NULL) {
            if (temp->key == key) {
                return temp->value;
            }
            temp = temp->next;
        }
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int index = key % size;
        node* temp = dict[index];
        node* last_node = temp;;
        while(temp != NULL) {
            if (temp->key == key) {
                last_node->next = temp->next;
                return;
            }
            last_node = temp;
            temp = temp->next;
        }
    }
};

/*
709. 转换成小写字母

实现函数 ToLowerCase()，该函数接收一个字符串参数 str，并将该字符串中的大写字母转换成小写字母，之后返回新的字符串。

示例 1：
输入: "Hello" 输出: "hello"
*/
string toLowerCase(string str) {
	for(auto it=str.begin();it!=str.end();++it){
		if(*it>'A' && *it<'Z'){
			*it=*it-'A'+'a';
		}
	}

	return str;
}

/*
717. 1比特与2比特字符

有两种特殊字符。第一种字符可以用一比特0来表示。第二种字符可以用两比特(10 或 11)来表示。
现给一个由若干比特组成的字符串。问最后一个字符是否必定为一个一比特字符。给定的字符串总是由0结束。

示例 1:
输入: 
bits = [1, 0, 0]
输出: True
解释: 唯一的编码方式是一个两比特字符和一个一比特字符。所以最后一个字符是一比特字符。
*/
bool isOneBitCharacter(vector<int>& bits) {
	        int len=bits.size();
        if(bits[len-1]==1) return false;  //最后一位是1,false
        for(int i=0;i<len;i++){
            if(i==len-1) return true;       //最后一位是0，就true（这里由于已经排除了最后一位是1的情况，只要可以到最后一位，就是对的）
            if(bits[i]==1&&i<len-1) i++; //其实10和11是一样的，都占两位，根本没有其他情况
        }
        return false;
}

/*
720. 词典中最长的单词

给出一个字符串数组words组成的一本英语词典。从中找出最长的一个单词，该单词是由words词典中其他单词逐步添加一个字母组成。
若其中有多个可行的答案，则返回答案中字典序最小的单词。若无答案，则返回空字符串。

示例 1:
输入: 
words = ["w","wo","wor","worl", "world"]
输出: "world"
解释: 
单词"world"可由"w", "wo", "wor", 和 "worl"添加一个字母组成。
*/
string longestWord(vector<string>& words) {
	        sort(words.begin(),words.end());
        set<string> tmp;
        tmp.insert("");
        string res="";
        for(auto word:words)
        {
            auto cc=word.substr(0,word.length()-1);
            if(tmp.find(cc)!=tmp.end())
            {
                tmp.insert(word);
                if(word.length()>res.length())
                {
                    res=word;
                }
            }
        }
        return res;
}

/*
724. 寻找数组的中心索引

给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。
我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1:
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 索引3 (nums[3] = 6) 的左侧数之和(1 + 7 + 3 = 11)，与右侧数之和(5 + 6 = 11)相等。
同时, 3 也是第一个符合要求的中心索引。
*/
int pivotIndex(vector<int>& nums) { 
        if (nums.size() == 0) return -1;
        int left=0;
        int right = 0;
        for (size_t i=1;i<nums.size();i++){
            right += nums[i];
        }
        for (size_t i=0;i<nums.size() -1 ;i++){
            if (right == left) return i;
            right = right - nums[i+1];
            left += nums[i];
        }
        if (left==right) return nums.size() - 1;
        else return -1;
}

/*
728. 自除数

自除数 是指可以被它包含的每一位数除尽的数。
例如，128 是一个自除数，因为 128 % 1 == 0，128 % 2 == 0，128 % 8 == 0。
还有，自除数不允许包含 0 。
给定上边界和下边界数字，输出一个列表，列表的元素是边界（含边界）内所有的自除数。

示例 1：
输入： 
上边界left = 1, 下边界right = 22
输出： [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
*/
vector<int> selfDividingNumbers(int left, int right) {
	std::vector<int> res;
	for(int i=left;i<=right;++i){
		int num=i;
		while(num !=0){	
			
			int tmp=num%10;	
			if(tmp==0 ||(tmp!=0 && i%tmp!=0) ){
			
				break;
			}

			num=num/10;
		}
		
		if(num==0){
			res.push_back(i);
		}
	}

	return res;
}

/*
733. 图像渲染

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。
给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。
为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，
接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。
将所有有记录的像素点的颜色值改为新的颜色值。最后返回经过上色渲染后的图像。

示例 1:
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。
*/
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
	if(image.size()==0)return image;
        int dx[4]={-1,0,1,0},dy[4]={0,1,0,-1};
        int oldColor=image[sr][sc];//记录旧颜色
        image[sr][sc]=newColor;//变新颜色
        if(oldColor==newColor)return image;//特判一下不然会死循环
        for(int i=0;i<4;i++)
        {
            int x=sr+dx[i],y=sc+dy[i];
            if(x>=0 && x<image.size() && y>=0 &&y<image[0].size()&&image[x][y]==oldColor)
            {
                floodFill(image,x,y,newColor);
            }
        
        }

        return image;
}

/*
744. 寻找比目标字母大的最小字母

给定一个只包含小写字母的有序数组letters 和一个目标字母 target，寻找有序数组里面比目标字母大的最小字母。
数组里字母的顺序是循环的。举个例子，如果目标字母target = 'z' 并且有序数组为 letters = ['a', 'b']，则答案返回 'a'。

示例:
输入:
letters = ["c", "f", "j"]
target = "a"
输出: "c"
*/
char nextGreatestLetter(vector<char>& letters, char target) {
         int l = 0, r = letters.size() - 1;
        if(target >= letters[r] || target < letters[l])  return letters[l];
        while(l + 1 < r){ 
            int mid = l + (r - l)/2;
            if(letters[mid] > target) r = mid;
            else    l = mid;
        }
        return letters[r];
}

/*
746. 使用最小花费爬楼梯

数组的每个索引做为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

示例 1:
输入: cost = [10, 15, 20]
输出: 15
解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
*/
int minCostClimbingStairs(vector<int>& cost) {
	int p=0,q=0;
	int res=0;
	for(size_t i=0;i<cost.size()-1;++i){
		res=min(p+cost[i],q+cost[i+1]);
		p=q;
		q=res;
	}

	return res;
}

/*
747. 至少是其他数字两倍的最大数

在一个给定的数组nums中，总是存在一个最大元素 。
查找数组中的最大元素是否至少是数组中每个其他数字的两倍。
如果是，则返回最大元素的索引，否则返回-1。

示例 1:
输入: nums = [3, 6, 1, 0]
输出: 1
解释: 6是最大的整数, 对于数组中的其他整数,
6大于数组中其他元素的两倍。6的索引是1, 所以我们返回1.
*/
int dominantIndex(vector<int>& nums) {
	if(nums.empty()){
		return -1;
	}

	if(nums.size()==1){
		return 0;
	}

    map<int,int,greater<int>> maps;
	for(size_t i=0;i<nums.size();++i){
		maps.insert(make_pair(nums[i],i));
	}

	auto it=maps.begin();
    auto next=maps.begin();
    ++next;
	if(it->first > 2*next->first){
		return it->second;
	}else{
		return -1;
	}
}

/*
748. 最短完整词

如果单词列表（words）中的一个单词包含牌照（licensePlate）中所有的字母，那么我们称之为完整词。在所有完整词中，
最短的单词我们称之为最短完整词。单词在匹配牌照中的字母时不区分大小写，比如牌照中的 "P" 依然可以匹配单词中的 "p" 字母。
我们保证一定存在一个最短完整词。当有多个单词都符合最短完整词的匹配条件时取单词列表中最靠前的一个。
牌照中可能包含多个相同的字符，比如说：对于牌照 "PP"，单词 "pair" 无法匹配，但是 "supper" 可以匹配。

示例 1：
输入：licensePlate = "1s3 PSt", words = ["step", "steps", "stripe", "stepple"]
输出："steps"
说明：最短完整词应该包括 "s"、"p"、"s" 以及 "t"。
对于 "step" 它只包含一个 "s" 所以它不符合条件。同时在匹配过程中我们忽略牌照中的大小写。
*/
string shortestCompletingWord(string licensePlate, vector<string>& words) {
     map<char,int> temp1;
    for(size_t i=0;i<licensePlate.size();++i){

        if(licensePlate[i]>='A' && licensePlate[i]<='Z'){
            temp1[licensePlate[i]+32]++;
        }else if(licensePlate[i]>='a' && licensePlate[i]<='z'){
            temp1[licensePlate[i]]++;
        }else{
            continue;
        }
    }

    int min=INT_MAX;
    int index=0;
    for(size_t i=0;i<words.size();++i){
        map<char,int> temp2;
        for(size_t j=0;j<words[i].size();++j){
            temp2[words[i][j]]++;
        }

        auto it=temp1.begin();
        for(;it!=temp1.end();++it){
            if(temp2[it->first]<it->second){
                break;
            }
        }

        if(it==temp1.end()){
            if(min>(int)words[i].size()){
                min=(int)words[i].size();
                index=i;
            }
        }
    }

    return words[index];
}

/*
762. 二进制表示中质数个计算置位

给定两个整数 L 和 R ，找到闭区间 [L, R] 范围内，计算置位位数为质数的整数个数。
（注意，计算置位代表二进制表示中1的个数。例如 21 的二进制表示 10101 有 3 个计算置位。还有，1 不是质数。）

示例 1:
输入: L = 6, R = 10
输出: 4
解释:
6 -> 110 (2 个计算置位，2 是质数)
7 -> 111 (3 个计算置位，3 是质数)
9 -> 1001 (2 个计算置位，2 是质数)
10-> 1010 (2 个计算置位，2 是质数)
*/
int countPrimeSetBits(int L, int R) {
	int iCount=0;
    for(int i=L;i<=R;++i){
        int num=i;
        int count=0;
        while(num){
            if(num&1)
                ++count;
            num=num>>1;
        }

        if(count==1)
            continue;
        //两个较小数另外处理
        if(count ==2|| count==3 )
            iCount++ ;
        //不在6的倍数两侧的一定不是质数
        if(count %6!= 1&&count %6!= 5)
            continue;
        int tmp =sqrt( count);
        //在6的倍数两侧的也可能不是质数
        for(int i= 5;i <=tmp; i+=6 )
            if(count %i== 0||count %(i+ 2)==0 )
                return 0 ;
        //排除所有，剩余的是质数
        iCount++;
    }

    return iCount;
}

/*
766. 托普利茨矩阵

如果一个矩阵的每一方向由左上到右下的对角线上具有相同元素，那么这个矩阵是托普利茨矩阵。
给定一个 M x N 的矩阵，当且仅当它是托普利茨矩阵时返回 True。

示例 1:
输入: 
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
输出: True
解释:
在上述矩阵中, 其对角线为:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。
各条对角线上的所有元素均相同, 因此答案是True。
*/
bool isToeplitzMatrix(vector<vector<int>>& matrix) {
	       if(matrix.empty()) return true;
        for(int i = 0; i < matrix[0].size(); i++){
            int pos_i = 0, pos_j = i, val = matrix[0][i];
            while(++pos_i < matrix.size() && ++pos_j < matrix[0].size()) {
                if(val != matrix[pos_i][pos_j]) return false;
            }
        }
        for(int i = 0; i < matrix.size(); i++) {
            int pos_i = i, pos_j = 0, val = matrix[i][0];
            while(++pos_i < matrix.size() && ++pos_j < matrix[0].size()) {
                if(val != matrix[pos_i][pos_j]) return false;
            }
        }
        return true;
}

/*
771. 宝石与石头

 给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 
 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

示例 1:
输入: J = "aA", S = "aAAbbbb"
输出: 3
*/
int numJewelsInStones(string J, string S) {
	map<char,int> maps;
	for(size_t i=0;i<J.size();++i){
		maps[J[i]]=0;
	}

	int iCount=0;
	for(size_t i=0;i<S.size();i++){
		if(maps.count(S[i])!=0){
			iCount++;
		}
	}

	return iCount;
}

/*
783. 二叉搜索树结点最小距离

给定一个二叉搜索树的根结点 root, 返回树中任意两节点的差的最小值。

示例：
输入: root = [4,2,6,1,3,null,null]
输出: 1
解释:
注意，root是树结点对象(TreeNode object)，而不是数组。
给定的树 [4,2,6,1,3,null,null] 可表示为下图:
          4
        /   \
      2      6
     / \    
    1   3  
最小的差值是 1, 它是节点1和节点2的差值, 也是节点3和节点2的差值。
*/
int minDiffInBST(TreeNode* root) {
	if(root==NULL){
		return INT_MAX;
	}

	int minValue=min(root->val-root->left->val,
		root->right->val-root->val);

	int left=minDiffInBST(root->left);
	int right=minDiffInBST(root->right);

	if(minValue<left && minValue<right) return minValue;
    if(left<minValue && left<right) return left;
    return right;
}

int main(){
	// cout<<hasAlternatingBits(1431655764)<<endl;

	// vector<int> res=selfDividingNumbers(1,22);

	string plate="1s3 PSt";
	vector<string> words={"step", "steps", "stripe", "stepple"};
	shortestCompletingWord(plate,words);
	return 0;
}