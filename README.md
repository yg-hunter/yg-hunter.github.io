# [leetcode analysis](#leetcode-analysis)
<br/>
<br/>

<h1 id="0">TABEL OF CONTENTS</h1>  

- ## 动态规划专题 Dynamic Programing
	1. [线性DP](#1)
		- [300.  最长上升子序列 LIS](#1.1)
		- [1143. 最长公共子序列 LCS](#1.2)
        - [120.  三角形最小路径和](#1.3)  
        - [53.   最大子序和](#1.4)   
        - [152.  乘积最大子数组](#1.5)  

  <br/>
  <br/>
  <br/>
  
***  
<h1 id="1.1"> LeetCode 300 </h1>  [回到目录](#0)  
## 1 [最长上升子序列 longest increasing subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence)

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

### 1.2.1 动态规划
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


### 1.2.2 改进版
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
  
  <br/>
  <br/>
  <br/>
  
***
<h1 id="1.2">LeetCode 1143</h1>  [回到目录](#0)  
## 2 [最长公共子序列 longest common subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)

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
|---|---|--|--|--|--|--|
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


### 2.3.2 动态规划
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
  <br/>
  <br/>
  <br/>
  
***
<h1 id="1.3">LeetCode 120</h1>  [回到目录](#0)  
## 3 [三角形最小路径和 triangle](https://leetcode-cn.com/problems/triangle/)   
## 3.1 题目描述
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

#### 示例:
```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。

#### 说明:
如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。

## 3.2 解题思路
1、暴力枚举法，找出所有路径并计算出他们的和，找出最小的即可，这样的话复杂度就是O(2的N次方)。

2、可以尝试用贪心法，本题示例好像是没问题，但是我们仔细想想就感觉不对了，看下面示例：  
```
[
      [2],
    [4, 8],
   [2, 3, 1],
  [70,60,15,6]
]
```  
按照贪心的思路，就是2+4+2+60=68，显然不对，为什么呢？因为本题中我们前面选的元素位置，就决定了后面我们只能选哪些数据。故贪心法不适合。  
  
3、动态规划解法，对于动态规划解法一般都是从后往前推的思路，跟第一种回溯法思路反过来。下面我们就来重点分析下动态规划。  


  
## 3.3 解决方案  
  
### 3.3.1 动态规划
DP状态定义：DP[i][j]表示从底端往上到第i行第j列时，走过的路径和的最小值。  
状态转移方程：DP[i][j]=min(DP[i+1][j],DP[i+1][j+1])+triagle[i][j];  
因为是至底向上找，所以初始值就是最后一行的元素的值，即DP[m-1][j]=triangle[m-1][j]，最后路径和的最小值就是在DP[0][0]中。  
该算法复杂度为O(m*n)，空间复杂度也是O(m*n)。

c++代码实现如下：  

```
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        if(triangle.empty())
            return 0;

        int m = triangle.size();
    	if(m == 1)
    		return triangle[0][0];

    	int n = triangle[m-1].size();
    	vector<vector<int>> DP(m, vector<int>(n,0));

    	DP[m-1] = triangle[m-1]; //初始化最后一行

    	for (int i = m-2; i >= 0; i--){ //从倒数第二行开始
    		for (int j = 0; j < triangle[i].size(); j++){
    			DP[i][j] = min(DP[i+1][j], DP[i+1][j+1]) + triangle[i][j];
    		}
    	}

    	return DP[0][0];
    }
};
```
上面代码可以改进下，只用一个一维数组存放DP状态，因为其实在计算过程中，始终只使用了其中的一层状态，完全可以复用的，代码如下：
```
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size();
    	vector<int> DP(m, 0);

    	DP = triangle[m-1]; //初始化最后一行

    	for (int i = m-2; i >= 0; i--){ //从倒数第二行开始
    		for (int j = 0; j < triangle[i].size(); j++){
    			DP[j] = min(DP[j], DP[j+1]) + triangle[i][j];
    		}
    	}

    	return DP[0];
    }
};
```

我把上面c++代码改成了c代码之后，提交leetcode,执行用时降低了1/3，可见还是c代码效率高啊
```
int min(int x, int y){
    return (x<y)?x:y;
}

int minimumTotal(int** triangle, int triangleSize, int* triangleColSize){
    if(triangle == NULL || *triangle == NULL || triangleSize == 0 || *triangleColSize == NULL)
        return 0;

    if(triangleSize == 1)
        return triangle[0][0];
    int *DP = (int*)malloc(triangleColSize[triangleSize-1]*sizeof(int));
    for(int i = 0; i < triangleColSize[triangleSize-1]; i++)
        DP[i] = triangle[triangleSize-1][i];

    for (int i = triangleSize-2; i >= 0; i--){ //从倒数第二行开始
        for (int j = 0; j < triangleColSize[i]; j++){
            DP[j] = min(DP[j], DP[j+1]) + triangle[i][j];
        }
    }

    int min_len = DP[0];
    free(DP);
    return min_len;
}
```  

### 3.3.2  其它解法  

题解中看到一种巧妙的解法，因为本题要求的是每个数下面一层，只能取相邻的俩数，所以我们可以跟动态规划相同思路，从底往上，相邻的两个数取最小的，加到上面一层，逐层这样操作，最后最顶端的元素triangel[0][0]的值，即为路径和的最小值。  
c++代码如下：  
```
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
    	if(triangle.empty())
    		return 0;

    	for (int i = triangle.size()-1; i > 0; i--){ 
    		for (int j = 0; j < triangle[i].size()-1; j++){
    			triangle[i-1][j] = min(triangle[i][j], triangle[i][j+1]) + triangle[i-1][j];
    		}
    	}

    	return triangle[0][0];
    }
};
```  
  
  <br/>
  <br/>
  <br/>
  
<h1 id="1.4"> LeetCode 53 </h1>  [回到目录](#0)  
## 4 [最大子序和 maximum subarray](https://leetcode-cn.com/problems/maximum-subarray/)

## 4.1 题目描述
给定一个整数数组`nums`，找到一个具有最大和的连续子数组(子数组最少包含一个元素)，返回其最大和。

#### 示例:
```
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```  

**进阶:** 如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解  

## 4.2 解题思路
① 暴力枚举法，找出所有连续子序列并计算出他们的和，找出最大的即可，这样的话复杂度就是O(2的N次方)。
② 动态规划解法，做过了前面几道题，这题就相对简单点了。   
下面看下动态规划解法。  
  
  

## 4.3 解决方案  

### 4.3.1 动态规划  
分两步走：  
1 状态的定义：定义DP[i]，表示第i个元素及其前面元素组成的子数组(下标0-i)中，连续子序列的和。  
2 状态转移方程：DP[i]=max(DP[i], DP[i-1])+nums[i],初始DP[0] = nums[0]。  

我们可以用一个临时变量存储到i处时当前最大的连续子序列和，遍历完数组后，临时变量的值就是要求的值，这里只需要一层循环，故复杂度为O(N).  


c++代码如下：
```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.empty())
    		return 0;

    	int len = nums.size();
    	vector<int> DP(len);
    	DP[0] = nums[0];
    	int res = DP[0];

    	for(int i = 1; i < len; i++){
    		DP[i] = max(DP[i], DP[i-1]) + nums[i];
    		res = max(res, DP[i]);
    	}
    	
    	return res; 
    }
};
```
对应的c代码如下：
```
int max(int x, int y){
    return (x>y)?x:y;
}

int maxSubArray(int* nums, int numsSize){
	if(nums == NULL || numsSize <= 0)
		return 0;
    
    int *DP = (int*)malloc(numsSize*(sizeof(int)));
    memset(DP, 0, numsSize*(sizeof(int)));
	DP[0] = nums[0];

	int res = DP[0];
	for(int i = 1; i < numsSize; i++){
		DP[i] = max(DP[i], DP[i-1]) + nums[i];
		res = max(res, DP[i]);
	}
    free(DP);
	return res;
}
```

### 4.3.2 其它解法
看到题解中有个方法挺好，这里也说下，他的思路是这样的：
- 对数组进行遍历，当前最大连续子序列和为`sum`，结果为`ans`.  
- 如果`sum > 0`，则说明`sum`对结果有增益效果，则`sum`保留并加上当前遍历数字.  
- 如果`sum <= 0`，则说明`sum`对结果无增益效果，需要舍弃，则`sum`直接更新为当前遍历数字.  
- 每次比较`sum`和`ans`的大小，将最大值置为`ans`，遍历结束返回结果.  
这样只需额外几个变量即可，也节省了内存。  

我把它写成c代码如下：
```
int max(int x, int y){
    return (x>y)?x:y;
}

int maxSubArray(int* nums, int numsSize){
	if(nums == NULL || numsSize <= 0)
		return 0;
    
    int res = nums[0];
    int sum = 0;
    for(int i = 0; i < numsSize; i++) {
        if(sum > 0)
            sum += nums[i];
        else
            sum = nums[i];

        res = max(res, sum);
    }
    return res;
}
```
  <br/>
  <br/>
  <br/>
  
<h1 id="1.5"> LeetCode 152 </h1>  [回到目录](#0)  
## 5 [乘积最大子数组 maximum product subarray](https://leetcode-cn.com/problems/maximum-product-subarray/)  


## 5.1 题目描述
给你一个整数数组`nums`，请你找出数组中乘积最大的连续子数组(该子数组中至少包含一个数字)。

#### 示例 1:
```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```    
#### 示例 2:
```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```  

## 5.2 解题思路
① 暴力枚举法，也叫递归法，这里就不看了。  
② 动态规划解法，跟上面那题子序列和最大类似，不过比它复杂点。  
下面具体看下解法。  
  
  

## 5.3 解决方案  

### 5.3.1 动态规划  
因为这里整数数组里可能有为负数的值，而且负数乘以负数又会是正的值，那么我们不能像上面那题求子数组和最大的求法了，这里对于每个nums[i]，要计算正的最大值，也要计算负的最小值，因为后面如果有负数的话，负的最小值乘以负数，就会是正的最大值了。  
分两步走：  
1 状态的定义：定义DP[2][i][，DP[0]里存的都是正的最大值，DP[1]里存的都是负的最小值；DP[0][i]表示第i个元素及其前面元素组成的子数组(下标0-i)中，连续子序列乘积的最大值，DP[1][i]表示第i个元素及其前面元素组成的子数组(下标0-i)中，连续子序列乘积的负最小值。  
2 状态转移方程：  


我们可以用一个临时变量存储到i处时当前最大的连续子序列和，遍历完数组后，临时变量的值就是要求的值，这里只需要一层循环，故复杂度为O(N). 

c++代码如下，这里我用了俩数组，效果跟DP[2][i]一样：  
```
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        if(nums.empty())
    		return 0;

    	int len = nums.size();

    	vector<int> max_val(len);
    	vector<int> min_val(len);
    	max_val[0] = nums[0];
    	min_val[0] = nums[0];
    	int max_product = nums[0];	

    	for(int i = 1; i < len; i++){
    		max_val[i] = max(max(max_val[i-1]*nums[i], min_val[i-1]*nums[i]), nums[i]);
    		min_val[i] = min(min(max_val[i-1]*nums[i], min_val[i-1]*nums[i]), nums[i]);
    		max_product = max(max_product, max_val[i]);
    	}
	    
    	return max_product;
    }
};
```  

这里可以优化下，上面其实没必要用数组， 因为我们每次都取最小或最大值，所以可以只用两个变量存储，改进代码如下:  
  
```
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        if(nums.empty())
    		return 0;

    	int max_val = nums[0], min_val = nums[0], max_product = nums[0];	

    	for(int i = 1; i < nums.size(); i++){
    		max_val = max_val*nums[i];
    		min_val = min_val*nums[i];

    		max_val = max(max(max_val, min_val), nums[i]);
    		min_val = min(min(min_val, max_val), nums[i]);            
    		max_product = max(max_product, max_val);
    	}
	    
    	return max_product;
    }
};
```
