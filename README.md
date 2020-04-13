# <a name="LeetCode_300">LeetCode_300</a>
1 [最长上升子序列 longest increasing subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence)

## 1.1 题目描述
给定一个无序的整数数组，找到其中最长上升子序列的长度。

#### 示例:
```
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

#### 说明:
* 可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。  
* 你算法的时间复杂度应该为 O(n2)。    

**进阶:** 你能将算法的时间复杂度降低到 O(n log n) 吗? 

## 1.2 解题思路
看完题目描述，我还以为是要找的这个子序列是连续元素组成，直到看到示例，才发现这个子序列可以不连续。如果要求的子序列是连续的话，这题就简单多了，直接遍历一遍，比较相邻元素的大小即可。  
那么对于可以不连续的子序列来说，问题好像没那么简单了。  
首先能想到的最直接的方法，就是暴力穷举的方式，但是复杂度比较高，这里就不具体说了。  
下面看下动态规划解法。  

## 1.2.1 动态规划
我们再来分析下示例 [10,9,2,5,3,7,101,18] ，其实针对每个元素来说，它及之前元素组成的数组中的最长上升子序列都是确定了的，也就是说具有最优子结构，这样我们就可以从后往前逐步计算出整个数组的最长上升子序列。  
分两步走：  
1 状态的定义：定义DP[i]，表示第i个元素及其前面元素组成的子数组(下标0-i)中，最长的上升子序列的长度。
2 状态转移方程：在第i个元素之前的所有子数组中，对于j(j<i)，第j个元素及之前的元素组成的子数组中上升子序列长度就是DP[j]，那么对于nums[j]<nums[i]时,则DP[i]=max{DP[j]+1}，否则DP[i]不变。

这里做了两重循环，所以复杂度为O(N²)，空间复杂度也是O(N²)。

c++代码实现如下：  
```C++
int lengthOfLIS(vector<int>& nums) {
    int vec_len = nums.size();
    if(vec_len < 2) //数组长度为0，返回0；数组长度为1，则最长上升子序列就是1
        return vec_len;

    int max_len = 1;
    vector<int> DP(vec_len, 1); //存放第i个元素及之前的，所有上升子序列的长度，初始都是1

    for(int i = 1; i < vec_len; i++){ //逐个元素，计算其前面所有元素组成的数组中，最长的上升子序列长度
        for(int j = 0; j < i; j++){
            if(nums[j] < nums[i]){ //因为是按递增来算长度
                DP[i] = std::max(DP[i], DP[j] + 1); //第i个元素及其前面元素中，最长子序列长度就是
                                                          //取其前面元素中最大的+1与当前最大的长度中最大的
            }                 
            //else if(nums[j] >= nums[i]){} 这里就不是上升序列了，所以长度不变
        }
        max_len = std::max(max_len, DP[i]);
    }

    return max_len;
}
```   


## 1.2.2 改进版
这个是我看到题解中的解法，很巧妙的解法，它把上面动态规划的解法中，第二重循环做了优化。它是维护了一个数组，用来暂存了可能成为最长上升子序列的所有元素，最后这个数组的长度就是要求的最长上升子序列的长度，具体做法如下：
逐个的取出原数组nums中元素，往临时数组tmp_lis中放，若该元素a大于tmp_lis中所有元素，则追加到tmp_lis最后；若不是，则找出tmp_lis中大于该元素a的最小的元素b，然后用a替换tmp_lis中的这个元素b，直到nums数组遍历完毕。

我们来看下，对于示例数组：
[10,9,2,5,3,7,101,18]
它是逐个的遍历数组元素，往另一个数组中放，假设这个数组为tmp_lis[]。  

取第一个元素10：  
tmp_lis[] = {10}  
取第二个元素9，因为9小于10，则用9替换10：  
tmp_lis[] = {9}  
取第三个元素2，因为2  
tmp_lis[] = {2}  
取第四个元素5，因为5>2,所以追加到tmp_lis中   
tmp_lis[] = {2,5}  
取第五个元素3，因为2<3<5,所以用3替换5  
tmp_lis[] = {2,3}  
取第六个元素7，因为7>3>2,所以直接追加到tmp_lis中  
tmp_lis[] = {2,3,7}  
取第七个元素101，因为101>7>3>2,所以直接追加到tmp_lis中  
tmp_lis[] = {2,3,7,101}  
取第八个元素18，因为18<101,18>7,所以用18替换101  
tmp_lis[] = {2,3,7,18}  
所以最终最长上升子序列的长度就是tmp_lis的长度，为4。  

对于查找tmp_lis中比nums[i]大的最小元素，我们采用二分法查找，所以这层循环的复杂度是O(logN)，加上外层的O(N)的复杂度，最终算法复杂度就是O(NlogN)。  
**注意:** tmp_lis最后并非就是最长上升子序列，它只是暂存了曾经是最长上升子序列的数组，可以自行以[6,7,8,9,1,2,3]为例自行验算下。

c++代码实现如下：  
```C++
int lengthOfLIS(vector<int>& nums) {
    vector<int> tmp_lis;
    for(int i = 0; i < nums.size(); i++){
        auto it = std::lower_bound(tmp_lis.begin(), tmp_lis.end(), nums[i]);/二分查找
        if(it != tmp_lis.end())
            *it = nums[i];
        else
            tmp_lis.push_back(nums[i]);//因为tmp_lis已经有序，所以找不到比*it大的数据的时候，则*it比tmp_lis中所有元素都大，直接追加
    }

    return tmp_lis.size();
}
```   
  
  
  

# <a name="LeetCode_1143">LeetCode_1143</a>
# 2 [最长公共子序列 longest common subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/submissions/)

## 2.1 题目描述
给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长公共子序列。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回 0。

#### 示例 1:
```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

#### 示例 2:
```
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。
```

#### 提示:
* 1 <= text1.length <= 1000  
* 1 <= text2.length <= 1000
* 输入的字符串只含有小写英文字符。    


## 2.2 解题思路
首先最直观的方法还是可以暴力穷举法，找出text1、text2的所有子序列，然后逐个判断是否匹配，最后找到最长的匹配的子序列即得到其长度。这种方法复杂度比较高，不去看了。

再来仔细分析下，看下如下示例，text1="bcdefg",text2="cbfdg",建立二维表如下，当行列对应的字符相同时，对应的值置1。  
  
  
|  | b | c | d | e |f|g|
|-------------|
| c |   | 1 |||||
| b |1 | || |||
|f|||||1||
|d|||1||||
|g||||||1||  
  
可以看出，将这个二维表遍历，当我们只向下或向右交替着遍历(为了保持字符在原串中的顺序)时，遇到值为1的就取出对应字符，最终组成的字符串就是公共子序列(bdg,bfg、cdg,cfg)，故而就能得到最长公共子序列了。之后对这个二维数组按上述方法找出1最多的就是答案了，此解法还有很大优化空间，比如下面的DP解法。

还有就是动态规划的解法了，我们深入分析题目之后，会发现本题跟LeetCode 300题的最长上升子序列问题有点类似，到两个字符串的某一个字符的时候，其实其最长公共子序列的长度是确定的，存在最优子结构，可以从前面的字符串逐步往后推导出最后的最优解，其实也是对上面那个解法的优化。下面就来重点看下动态规划解法。 

## 2.3 解决方案
对于动态规划，首先还是两步走：
1 定义状态DP[i][j]，表示字符串text1[0..i]跟text2[0...j]之间拥有的最长公共子序列的长度。
2 状态转移方程：
	可以有三种状态到达DP[i][j]，分别为DP[i-1][j-1]、DP[i-1][j]、DP[i][j-1]，对于这三个前序状态：
    如果text1[i]==text2[j]的话，DP[i][j]=DP[i-1][j-1]+1，其它情况则，DP[i][j]取的应该是DP[i-1][j]、DP[i][j-1]中最大的。

下面我们看下具体代码实现。

### 2.3.1 递归法
根据我们上面动态规划里面的递推公式，其实可以用递归方法解决，代码如下：
```C++
int lcs(string &t1, string &t2, int len1, int len2)
{
    if (len1 == 0 || len2 == 0)
        return 0;

    if (t1[len1-1] == t2[len2-1])
        return lcs(t1, t2, len1-1, len2-1) + 1;
    else
        return max(lcs(t1, t2, len1, len2-1), lcs(t1, t2, len1-1, len2));
}
int longestCommonSubsequence(string text1, string text2) {
    return lcs(text1, text2, text1.size(), text2.size());
}
```



## 2.2.2 动态规划
上面递归法中存在大量重叠计算问题，时间及空间复杂度比较高，下面动态规划解法优化了上面求解过程，代码如下：
```C++
int longestCommonSubsequence(string text1, string text2) {
    if(text1.empty() || text2.empty())
        return 0;
    
    int len1 = text1.size();
    int len2 = text2.size();
    vector<vector<int>> DP(len1+1, vector<int>(len2+1, 0)); //二维数组初始化为0

    for(int i = 1; i <= len1; i++){
        for(int j = 1; j <= len2; j++){
            if(text1[i-1] == text2[j-1])
                DP[i][j] = DP[i-1][j-1] + 1;
            else
                DP[i][j] = max(DP[i-1][j], DP[i][j-1]);
        }
    } 

    return DP[len1][len2];
}
```   

