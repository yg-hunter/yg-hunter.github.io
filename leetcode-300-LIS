[300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence) longest increasing sequence

###题目描述
给定一个无序的整数数组，找到其中最长上升子序列的长度。

###示例:
```
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

###说明:
-可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
-你算法的时间复杂度应该为 O(n2) 。
**进阶:** 你能将算法的时间复杂度降低到 O(n log n) 吗?

##动态规划


```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int vec_len = nums.size();
        if(vec_len < 2)
            return vec_len;
        
        int max_len = 1;
        vector<int> lens(vec_len, 1);
    
        for(int i = 1; i < vec_len; i++){
            for(int j = 0; j < i; j++){
                if(nums[j] < nums[i])
                    lens[i] = std::max(lens[i], lens[j] + 1);
            }
            max_len = std::max(max_len, lens[i]);
        }

        return max_len;
    }
};
```
