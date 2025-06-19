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
              <el-input v-model="currentUid" size="small"></el-input>
              <el-button type="primary" size="small" @click="fetchLearnerData">确定</el-button>
            </div>
          </div>
          <div class="info-item">
            <span>Email:</span>
            <span>{{ learnerInfo.email }}</span>
          </div>
          <div class="info-item">
            <span>手机号:</span>
            <span>{{ learnerInfo.phone }}</span>
          </div>
        </el-card>

        <!-- 学习领域 -->
        <el-card class="domains-card">
          <div slot="header">学习过的领域</div>
          <div class="domains-list">
            <el-tag
              v-for="domain in learnerInfo.domains"
              :key="domain.id"
              @click="selectDomain(domain)"
              :type="currentDomainId === domain.id ? 'primary' : ''"
            >
              {{ domain.name }}
            </el-tag>
          </div>
        </el-card>

        <!-- 总体表现 -->
        <el-card class="performance-card">
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
      <div class="right-panel">
        <!-- 领域表现 - 修改为左右布局 -->
        <el-card v-if="selectedDomain" class="domain-card">
          <div slot="header">{{ selectedDomain.name }}领域表现</div>
          <div class="domain-content">
            <!-- 知识点区域 (3/4宽度) -->
            <div class="knowledge-container">
              <div class="knowledge-grid">
                <div v-for="point in selectedDomain.knowledgePoints" :key="point.name" class="knowledge-item">
                  <el-tooltip :content="`${point.name}: 预测正确率:${(point.score)}`" placement="top">
                    <div class="point-container">
                      <span class="point-name" :title="point.name">{{ truncateName(point.name) }}</span>
                      <span :class="['point-score', getScoreClass(point.score)]"></span>
                    </div>
                  </el-tooltip>
                </div>
              </div>
            </div>
            
            <!-- 图表区域 (1/4宽度) -->
            <div class="chart-container">
              <div ref="domainChart" class="chart"></div>
              <div class="domain-evaluation">
                <el-tag :type="getEvaluationTagType(selectedDomain.evaluation)" size="medium">
                  {{ selectedDomain.evaluation }}
                </el-tag>
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
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch, nextTick } from 'vue';
import * as echarts from 'echarts';

export default {
  setup() {
    // 模拟数据
    const mockData = {
      uid: '10001',
      email: '10001@example.com',
      phone: '13800011234',
      domains: [
        {
          id: 'math',
          name: '数学',
          knowledgePoints: [
            { name: '代数基础', score: 0.85 },
            { name: '几何基础', score: 0.88 },
            { name: '概率统计', score: 0.92 }
          ]
        },
        {
          id: 'physics',
          name: '物理',
          knowledgePoints: [
            { name: '力学', score: 0.72 },
            { name: '电磁学', score: 0.75 },
            { name: '热学', score: 0.78 }
          ]
        },
        {
          id: 'chemistry',
          name: '化学',
          knowledgePoints: [
            { name: '无机化学', score: 0.55 },
            { name: '有机化学', score: 0.65 },
            { name: '物理化学', score: 0.58 }
          ]
        },
        {
          id: 'biology',
          name: '生物',
          knowledgePoints: [
            { name: '细胞生物学', score: 0.35 },
            { name: '遗传学', score: 0.45 },
            { name: '生态学', score: 0.38 }
          ]
        }
      ]
    };

    // 根据知识点评分计算领域评价
    mockData.domains.forEach(domain => {
      const scores = domain.knowledgePoints.map(p => p.score);
      const minScore = Math.min(...scores);
      
      if (minScore >= 0.8) {
        domain.evaluation = '表现优秀';
      } else if (minScore >= 0.6) {
        domain.evaluation = '表现良好';
      } else if (minScore >= 0.4) {
        domain.evaluation = '存在问题';
      } else {
        domain.evaluation = '学情预警';
      }
    });

    const mockRecommendations = {
      knowledgePoints: ['高级算法', '机器学习', '深度学习基础', '神经网络', '计算机视觉'],
      studyPartners: ['10002', '10003', '10005'],
      studyModels: ['10010', '10015', '10020']
    };

    // 响应式数据
    const currentUid = ref('10001');
    const learnerInfo = ref(JSON.parse(JSON.stringify(mockData)));
    const selectedDomain = ref(learnerInfo.value.domains[0]);
    const currentDomainId = ref(learnerInfo.value.domains[0].id);
    const recommendations = ref({ ...mockRecommendations });
    
    const performanceItems = ref([
      { type: 'excellent', label: '表现优秀', count: 0 },
      { type: 'good', label: '表现良好', count: 0 },
      { type: 'problem', label: '存在问题', count: 0 },
      { type: 'warning', label: '学情预警', count: 0 }
    ]);

    // 图表实例
    const overallChart = ref(null);
    const domainChart = ref(null);

    // 方法
    const fetchLearnerData = () => {
      // 模拟API调用
      console.log(`Fetching data for UID: ${currentUid.value}`);
      
      // 重置数据
      learnerInfo.value = JSON.parse(JSON.stringify(mockData));
      selectedDomain.value = learnerInfo.value.domains[0];
      currentDomainId.value = learnerInfo.value.domains[0].id;
      recommendations.value = { ...mockRecommendations };
      
      // 更新评价
      updatePerformanceSummary();
      
      // 重新渲染图表
      nextTick(() => {
        initCharts();
      });
    };

    const selectDomain = (domain) => {
      selectedDomain.value = domain;
      currentDomainId.value = domain.id;
      nextTick(() => {
        initDomainChart();
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
      // 重置计数器
      performanceItems.value.forEach(item => item.count = 0);
      
      // 重新统计
      learnerInfo.value.domains.forEach(domain => {
        if (domain.evaluation === '表现优秀') {
          performanceItems.value[0].count++;
        } else if (domain.evaluation === '表现良好') {
          performanceItems.value[1].count++;
        } else if (domain.evaluation === '存在问题') {
          performanceItems.value[2].count++;
        } else if (domain.evaluation === '学情预警') {
          performanceItems.value[3].count++;
        }
      });
    };

    const initOverallChart = () => {
      // 销毁旧图表
      if (overallChart.value) {
        overallChart.value.dispose();
      }
      
      // 初始化新图表
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
      
      // 设置空数据提示
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

    const initDomainChart = () => {
      if (!selectedDomain.value) return;
      
      // 销毁旧图表
      if (domainChart.value) {
        domainChart.value.dispose();
      }
      
      // 初始化新图表
      domainChart.value = echarts.init(document.querySelector('.domain-card .chart'));
      
      const points = selectedDomain.value.knowledgePoints;
      let excellent = 0, good = 0, problem = 0, warning = 0;
      
      points.forEach(point => {
        if (point.score >= 0.8) excellent++;
        else if (point.score >= 0.6) good++;
        else if (point.score >= 0.4) problem++;
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
      
      // 设置空数据提示
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
      
      domainChart.value.setOption(option);
    };

    const initCharts = () => {
      initOverallChart();
      initDomainChart();
    };

    const handleResize = () => {
      if (overallChart.value) overallChart.value.resize();
      if (domainChart.value) domainChart.value.resize();
    };

    // 生命周期
    onMounted(() => {
      updatePerformanceSummary();
      initCharts();
      window.addEventListener('resize', handleResize);
    });

    // 组件卸载时清理
    onUnmounted(() => {
      window.removeEventListener('resize', handleResize);
      if (overallChart.value) overallChart.value.dispose();
      if (domainChart.value) domainChart.value.dispose();
    });

    watch(selectedDomain, () => {
      initDomainChart();
    });

    return {
      currentUid,
      learnerInfo,
      selectedDomain,
      currentDomainId,
      recommendations,
      performanceItems,
      fetchLearnerData,
      selectDomain,
      getScoreClass,
      getEvaluationTagType,
      truncateName
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

/* 1. 左右面板1:2比例 */
.dashboard-container {
  display: flex;
  gap: 20px;
}

.left-panel {
  flex: 1; /* 1/3宽度 */
}

.right-panel {
  flex: 2; /* 2/3宽度 */
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

.domains-list .el-tag {
  margin-right: 10px;
  margin-bottom: 10px;
  cursor: pointer;
}

/* 总体表现图表 */
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

.excellent { background-color: #67C23A; } /* 绿色 */
.good { background-color: #409EFF; } /* 浅蓝色 */
.problem { background-color: #E6A23C; } /* 黄色 */
.warning { background-color: #F56C6C; } /* 红色 */

/* 3. 领域表现卡片左右布局 */
.domain-card {
  height: 380px; /* 固定高度以便滚动 */
}

.domain-content {
  display: flex;
  height: calc(100% - 57px); /* 减去标题高度 */
}

.knowledge-container {
  flex: 3; /* 3/4宽度 */
  overflow-y: auto; /* 垂直滚动 */
  padding-right: 10px;
}

.knowledge-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 8px;
}

.knowledge-item {
  display: flex;
  align-items: center;
  padding: 6px;
  background: #f5f7fa;
  border-radius: 4px;
  height: 32px;
}

.point-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.point-name {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 8px;
  font-size: 12px;
}

.point-score {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  flex-shrink: 0;
}

.chart-container {
  flex: 1; /* 1/4宽度 */
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding-left: 10px;
  border-left: 1px solid #ebeef5;
}

.domain-evaluation {
  margin-top: 10px;
  text-align: center;
}

/* 推荐内容样式 */
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

/* 响应式设计 */
@media (max-width: 992px) {
  .dashboard-container {
    flex-direction: column;
  }
  
  .domain-content {
    flex-direction: column;
  }
  
  .knowledge-container {
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

/* 美化滚动条 */
.knowledge-container::-webkit-scrollbar {
  width: 6px;
}

.knowledge-container::-webkit-scrollbar-thumb {
  background-color: #c1c1c1;
  border-radius: 3px;
}
</style>