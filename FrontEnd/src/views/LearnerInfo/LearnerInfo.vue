<template>
  <div class="learner-dashboard">
    <div class="header">
      <h1>学习者数据概览</h1>
      <p class="author">—— SZY</p>
    </div>
    
    <div class="dashboard-container">
      <!-- 左侧面板 (1/3宽度) -->
      <div class="left-panel">
        <!-- 学习者基本信息 -->
        <el-card class="info-card">
          <div slot="header">学习者基本信息</div>
          <div class="info-item">
            <span>UID:</span>
            <div class="uid-input-container">
              <el-input v-model="currentLrnUid" size="small"></el-input>
              <el-button type="primary" size="small" @click="fetchLearnerData" :loading="loading">确定</el-button>
            </div>
          </div>
          <div v-if="hasData" class="info-item">
            <span>Email:</span>
            <span>{{ learnerInfo.email }}</span>
          </div>
          <div v-if="hasData" class="info-item">
            <span>手机号:</span>
            <span>{{ learnerInfo.phone }}</span>
          </div>
          <div v-if="!hasData" class="empty-tip">
            <el-empty description="请输入UID查询学习者数据"></el-empty>
          </div>
        </el-card>

        <!-- 学习领域 -->
        <el-card class="areas-card" v-if="hasData">
          <div slot="header">学习过的领域</div>
          <div class="areas-list">
            <el-tag
              v-for="area in learnerInfo.are_data"
              :key="area.are_uid"
              @click="selectArea(area)"
              :type="currentAreUid === area.are_uid ? 'primary' : ''"
            >
              {{ area.are_name }}
            </el-tag>
          </div>
        </el-card>

        <!-- 总体表现 -->
        <el-card class="performance-card" v-if="hasData">
          <div slot="header">总体表现</div>
          <div ref="overallChart" class="chart"></div>
          <div class="performance-summary">
            <div v-for="item in performanceItems" :key="item.type" class="summary-item">
              <span :class="['indicator', item.type]"></span>
              <span>{{ item.label }}: {{ item.count }}个</span>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 右侧面板 (2/3宽度) -->
      <div class="right-panel" v-if="hasData">
        <!-- 领域表现 -->
        <el-card v-if="selectedArea" class="area-card">
          <div slot="header">{{ selectedArea.are_name }}领域表现</div>
          <div class="el-card__body" style="padding: 20px">
            <div class="area-content">
              <!-- 知识点区域 -->
              <div class="concept-container">
                <div class="concept-scroll-wrapper" @wheel.prevent="handleConceptScroll">
                  <div class="concept-grid">
                    <div v-for="cpt in selectedArea.cpt_data" :key="cpt.cpt_uid" class="cpt-item">
                      <el-tooltip :content="`${cpt.cpt_name}: 预测正确率:${(cpt.score)}`" placement="top">
                        <div class="cpt-container">
                          <span class="cpt-name" :title="cpt.cpt_name">{{ truncateName(cpt.cpt_name) }}</span>
                          <span :class="['cpt-score', getScoreClass(cpt.score)]"></span>
                        </div>
                      </el-tooltip>
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- 图表区域 -->
              <div class="chart-container">
                <div ref="areaChart" class="chart"></div>
                <div class="area-evaluation">
                  <el-tag :type="getEvaluationTagType(selectedArea.evaluation)" size="medium">
                    {{ selectedArea.evaluation }}
                  </el-tag>
                </div>
              </div>
            </div>
          </div>
          
        </el-card>

        <!-- 推荐内容 -->
        <el-card class="recommend-card">
          <div slot="header">推荐知识点</div>
          <div class="recommend-list">
            <el-tag 
              v-for="item in recommendations.knowledgePoints" 
              :key="item" 
              type="info"
              class="recommend-tag"
            >
              {{ item }}
            </el-tag>
          </div>
        </el-card>

        <el-card class="recommend-card">
          <div slot="header">推荐学习伙伴</div>
          <div class="partner-list">
            <div v-for="(partner, index) in recommendations.studyPartners" :key="partner" class="partner-item">
              <span>伙伴{{ index + 1 }}:</span>
              <el-tag size="small">{{ partner }}</el-tag>
            </div>
          </div>
        </el-card>

        <el-card class="recommend-card">
          <div slot="header">推荐学习榜样</div>
          <div class="partner-list">
            <div v-for="(model, index) in recommendations.studyModels" :key="model" class="partner-item">
              <span>榜样{{ index + 1 }}:</span>
              <el-tag type="success" size="small">{{ model }}</el-tag>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 空数据状态 -->
      <div class="right-panel" v-if="!hasData">
        <el-card class="empty-card">
          <el-empty description="暂无学习者数据"></el-empty>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch, nextTick } from 'vue';
import * as echarts from 'echarts';
import { getLearnerInfo, getRecommendations } from '@/api/learnerInfo.js';

export default {
  setup() {
    // 响应式数据
    const currentLrnUid = ref('');
    const learnerInfo = ref({
      lrn_uid: '',
      email: '',
      phone: '',
      are_data: []
    });
    const selectedArea = ref(null);
    const currentAreUid = ref('');
    const recommendations = ref({
      knowledgePoints: [],
      studyPartners: [],
      studyModels: []
    });
    const loading = ref(false);
    const hasData = ref(false);
    
    const performanceItems = ref([
      { type: 'excellent', label: '表现优秀', count: 0 },
      { type: 'good', label: '表现良好', count: 0 },
      { type: 'problem', label: '存在问题', count: 0 },
      { type: 'warning', label: '学情预警', count: 0 }
    ]);

    // 图表实例
    const overallChart = ref(null);
    const areaChart = ref(null);

    // 方法
    const fetchLearnerData = async () => {
      if (!currentLrnUid.value) {
        ElMessage.warning('请输入UID');
        return;
      }

      loading.value = true;
      try {
        // 获取学习者信息
        const learnerRes = await getLearnerInfo(currentLrnUid.value);
        if (learnerRes.data) {
          learnerInfo.value = learnerRes.data;
          
          // 计算领域评价
          learnerInfo.value.are_data.forEach(area => {
            const scores = area.cpt_data?.map(p => p.score) || [];
            const minScore = scores.length > 0 ? Math.min(...scores) : 0;
            
            if (minScore >= 0.8) {
              area.evaluation = '表现优秀';
            } else if (minScore >= 0.6) {
              area.evaluation = '表现良好';
            } else if (minScore >= 0.4) {
              area.evaluation = '存在问题';
            } else {
              area.evaluation = '学情预警';
            }
          });

          // 设置默认选中的领域
          if (learnerInfo.value.are_data.length > 0) {
            selectedArea.value = learnerInfo.value.are_data[0];
            currentAreUid.value = learnerInfo.value.are_data[0].are_uid;
          }

          // 获取推荐信息
          const recommendRes = await getRecommendations(currentLrnUid.value);
          recommendations.value = recommendRes.data || {
            knowledgePoints: [],
            studyPartners: [],
            studyModels: []
          };

          // 更新评价统计
          updatePerformanceSummary();
          hasData.value = true;
          
          // 渲染图表
          nextTick(() => {
            initCharts();
          });
        } else {
          hasData.value = false;
          ElMessage.warning('未找到该学习者的数据');
        }
      } catch (error) {
        console.error('获取学习者数据失败:', error);
        ElMessage.error('获取数据失败');
        hasData.value = false;
      } finally {
        loading.value = false;
      }
    };

    const selectArea = (area) => {
      selectedArea.value = area;
      currentAreUid.value = area.are_uid;
      nextTick(() => {
        initAreaChart();
      });
    };

    const getScoreClass = (score) => {
      if (score >= 0.8) return 'excellent';
      if (score >= 0.6) return 'good';
      if (score >= 0.4) return 'problem';
      return 'warning';
    };

    const getEvaluationTagType = (evaluation) => {
      switch(evaluation) {
        case '表现优秀': return 'success';
        case '表现良好': return 'primary';
        case '存在问题': return 'warning';
        case '学情预警': return 'danger';
        default: return '';
      }
    };

    const truncateName = (name) => {
      return name.length > 6 ? name.substring(0, 6) + '...' : name;
    };

    const updatePerformanceSummary = () => {
      performanceItems.value.forEach(item => item.count = 0);
      
      learnerInfo.value.are_data.forEach(area => {
        if (area.evaluation === '表现优秀') {
          performanceItems.value[0].count++;
        } else if (area.evaluation === '表现良好') {
          performanceItems.value[1].count++;
        } else if (area.evaluation === '存在问题') {
          performanceItems.value[2].count++;
        } else if (area.evaluation === '学情预警') {
          performanceItems.value[3].count++;
        }
      });
    };

    const handleConceptScroll = (e) => {
      const container = e.currentTarget;
      container.scrollTop += e.deltaY;
    };

    const initOverallChart = () => {
      if (overallChart.value) {
        overallChart.value.dispose();
      }
      
      overallChart.value = echarts.init(document.querySelector('.performance-card .chart'));
      
      const data = [
        { value: performanceItems.value[0].count, name: '表现优秀', itemStyle: { color: '#67C23A' } },
        { value: performanceItems.value[1].count, name: '表现良好', itemStyle: { color: '#409EFF' } },
        { value: performanceItems.value[2].count, name: '存在问题', itemStyle: { color: '#E6A23C' } },
        { value: performanceItems.value[3].count, name: '学情预警', itemStyle: { color: '#F56C6C' } }
      ].filter(item => item.value > 0);
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        series: [{
          name: '总体表现',
          type: 'pie',
          radius: ['50%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: false,
            position: 'center'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '18',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: false
          },
          data: data
        }]
      };
      
      if (data.length === 0) {
        option.graphic = {
          type: 'text',
          left: 'center',
          top: 'center',
          style: {
            text: '暂无数据',
            fontSize: 14,
            fill: '#999'
          }
        };
      }
      
      overallChart.value.setOption(option);
    };

    const initAreaChart = () => {
      if (!selectedArea.value) return;
      
      if (areaChart.value) {
        areaChart.value.dispose();
      }
      
      const chartDom = document.querySelector('.area-card .chart');
      if (!chartDom) return;
      
      areaChart.value = echarts.init(chartDom);
      
      const cpts = selectedArea.value.cpt_data || [];
      let excellent = 0, good = 0, problem = 0, warning = 0;
      
      cpts.forEach(cpt => {
        if (cpt.score >= 0.8) excellent++;
        else if (cpt.score >= 0.6) good++;
        else if (cpt.score >= 0.4) problem++;
        else warning++;
      });
      
      const data = [];
      if (excellent > 0) data.push({ value: excellent, name: '优秀', itemStyle: { color: '#67C23A' } });
      if (good > 0) data.push({ value: good, name: '良好', itemStyle: { color: '#409EFF' } });
      if (problem > 0) data.push({ value: problem, name: '不合格', itemStyle: { color: '#E6A23C' } });
      if (warning > 0) data.push({ value: warning, name: '严重不足', itemStyle: { color: '#F56C6C' } });
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        series: [{
          name: '知识点分布',
          type: 'pie',
          radius: ['50%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: false,
            position: 'center'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '18',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: false
          },
          data: data
        }]
      };
      
      if (data.length === 0) {
        option.graphic = {
          type: 'text',
          left: 'center',
          top: 'center',
          style: {
            text: '暂无数据',
            fontSize: 14,
            fill: '#999'
          }
        };
      }
      
      areaChart.value.setOption(option);
    };

    const initCharts = () => {
      initOverallChart();
      initAreaChart();
    };

    const handleResize = () => {
      if (overallChart.value) overallChart.value.resize();
      if (areaChart.value) areaChart.value.resize();
    };

    // 生命周期
    onMounted(() => {
      window.addEventListener('resize', handleResize);
    });

    onUnmounted(() => {
      window.removeEventListener('resize', handleResize);
      if (overallChart.value) overallChart.value.dispose();
      if (areaChart.value) areaChart.value.dispose();
    });

    watch(selectedArea, () => {
      initAreaChart();
    });

    return {
      currentLrnUid,
      learnerInfo,
      selectedArea,
      currentAreUid,
      recommendations,
      performanceItems,
      loading,
      hasData,
      fetchLearnerData,
      selectArea,
      getScoreClass,
      getEvaluationTagType,
      truncateName,
      handleConceptScroll
    };
  }
};
</script>

<style scoped>
.learner-dashboard {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.author {
  font-size: 14px;
  color: #999;
  text-align: right;
  margin-right: 20px;
}

.dashboard-container {
  display: flex;
  gap: 20px;
}

.left-panel {
  flex: 1;
}

.right-panel {
  flex: 2;
}

.el-card {
  margin-bottom: 20px;
}

.info-item {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.info-item span:first-child {
  width: 80px;
  font-weight: bold;
  flex-shrink: 0;
}

.uid-input-container {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.uid-input-container .el-input {
  flex: 1;
}

.areas-list .el-tag {
  margin-right: 10px;
  margin-bottom: 10px;
  cursor: pointer;
}

.chart {
  height: 200px;
  width: 100%;
}

.performance-summary {
  margin-top: 15px;
}

.summary-item {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.indicator {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 8px;
  flex-shrink: 0;
}

.excellent { background-color: #67C23A; }
.good { background-color: #409EFF; }
.problem { background-color: #E6A23C; }
.warning { background-color: #F56C6C; }

/* 重设卡片结构样式 */
.area-card {
  height: 380px; /* 或您需要的任何高度 */
  display: flex;
  flex-direction: column;
  
  /* 关键修改：穿透修改Element UI默认样式 */
  :deep(.el-card__body) {
    padding: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden; /* 防止内容溢出 */
  }
}

.area-content {
  flex: 1;
  min-height: 0;
  display: flex;
  margin: 0;
  /* 不再需要calc计算，flex:1会自动分配剩余空间 */
}

/* 确保知识点容器填满空间 */
.concept-container {
  flex: 3;
  height: 100%;
  overflow: hidden;
  position: relative;
  margin: 0;
  padding: 0;
  min-height: 0;
}

/* 确保滚动容器填满空间 */
.concept-scroll-wrapper {
  height: 100%;
  width: 100%;
  overflow-y: auto;
  padding-right: 10px;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  margin: 0;
}

/* 知识点网格 - 移除底部padding */
.concept-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 8px;
  padding: 8px 10px 0 10px; /* 只保留顶部padding */
  margin: 0;
  min-height: min-content;
}

.cpt-item {
  display: flex;
  align-items: center;
  padding: 6px;
  background: #f5f7fa;
  border-radius: 4px;
  height: 32px;
}

.cpt-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.cpt-name {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 8px;
  font-size: 12px;
}

.cpt-score {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  flex-shrink: 0;
}

/* 图表容器 - 精确控制高度 */
/* 调整图表容器 */
.chart-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding-left: 10px;
  border-left: 1px solid #ebeef5;
  height: 100%;
  min-width: 200px;
  margin: 0;
}

/* 图表区域高度固定 */
.chart-container .chart {
  height: 180px; /* 适当增加高度，填充更多空间 */
  margin: 0;
  padding: 0;
}

.area-evaluation {
  margin-top: 0; /* 改为0 */
  padding-top: 10px; /* 使用padding替代margin */
  text-align: center;
}

.recommend-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.recommend-tag {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.partner-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.partner-item {
  display: flex;
  align-items: center;
}

.partner-item span:first-child {
  width: 60px;
  flex-shrink: 0;
}

.empty-tip {
  padding: 20px 0;
}

.empty-card {
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

@media (max-width: 992px) {
  .dashboard-container {
    flex-direction: column;
  }
  
  .area-content {
    flex-direction: column;
  }
  
  .concept-container {
    flex: none;
    height: 60%;
    padding-right: 0;
  }
  
  .chart-container {
    flex: none;
    height: 40%;
    padding-left: 0;
    margin-top: 15px;
    border-left: none;
    border-top: 1px solid #ebeef5;
  }
}

.concept-scroll-wrapper::-webkit-scrollbar {
  width: 6px;
}

.concept-scroll-wrapper::-webkit-scrollbar-thumb {
  background-color: #c1c1c1;
  border-radius: 3px;
}

.concept-scroll-wrapper::-webkit-scrollbar-track {
  background-color: #f1f1f1;
}
</style>