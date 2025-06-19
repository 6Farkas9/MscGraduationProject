<template>
  <div class="learner-dashboard">
    <h1>学习者数据概览</h1>
    
    <div class="dashboard-container">
      <!-- 左侧面板 -->
      <div class="left-panel">
        <!-- 学习者基本信息 -->
        <el-card class="info-card">
          <div slot="header">学习者基本信息</div>
          <div class="info-item">
            <span>UID:</span>
            <el-input v-model="currentUid" size="small" @change="fetchLearnerData"></el-input>
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

      <!-- 右侧面板 -->
      <div class="right-panel">
        <!-- 领域表现 -->
        <el-card v-if="selectedDomain" class="domain-card">
          <div slot="header">{{ selectedDomain.name }}领域表现</div>
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
          <div ref="domainChart" class="chart"></div>
          <div class="domain-evaluation">
            <el-tag :type="getEvaluationTagType(selectedDomain.evaluation)" size="medium">
              {{ selectedDomain.evaluation }}
            </el-tag>
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
          evaluation: '表现优秀',
          knowledgePoints: [
            { name: '代数基础', score: 0.85 },
            { name: '几何基础', score: 0.78 },
            { name: '概率统计', score: 0.92 },
            { name: '微积分', score: 0.88 },
            { name: '线性代数', score: 0.82 },
            { name: '离散数学', score: 0.79 }
          ]
        },
        {
          id: 'physics',
          name: '物理',
          evaluation: '表现良好',
          knowledgePoints: [
            { name: '力学', score: 0.72 },
            { name: '电磁学', score: 0.65 },
            { name: '热学', score: 0.58 },
            { name: '光学', score: 0.62 },
            { name: '量子物理', score: 0.55 }
          ]
        },
        {
          id: 'programming',
          name: '编程',
          evaluation: '学情预警',
          knowledgePoints: [
            { name: 'Python基础', score: 0.75 },
            { name: '数据结构', score: 0.68 },
            { name: '算法设计', score: 0.35 },
            { name: '数据库', score: 0.42 },
            { name: '网络编程', score: 0.38 }
          ]
        }
      ]
    };

    const mockRecommendations = {
      knowledgePoints: ['高级算法', '机器学习', '深度学习基础', '神经网络', '计算机视觉'],
      studyPartners: ['10002', '10003', '10005'],
      studyModels: ['10010', '10015', '10020']
    };

    // 响应式数据
    const currentUid = ref('10001');
    const learnerInfo = ref({ ...mockData });
    const selectedDomain = ref(learnerInfo.value.domains[0]);
    const currentDomainId = ref(learnerInfo.value.domains[0].id);
    const recommendations = ref({ ...mockRecommendations });
    
    const performanceItems = ref([
      { type: 'excellent', label: '表现优秀', count: 1 },
      { type: 'good', label: '表现良好', count: 1 },
      { type: 'warning', label: '学情预警', count: 1 }
    ]);

    // 方法
    const fetchLearnerData = () => {
      // 这里应该是API调用，现在用模拟数据代替
      console.log(`Fetching data for UID: ${currentUid.value}`);
      learnerInfo.value = { ...mockData };
      selectedDomain.value = learnerInfo.value.domains[0];
      currentDomainId.value = learnerInfo.value.domains[0].id;
      recommendations.value = { ...mockRecommendations };
      updatePerformanceSummary();
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
      if (score >= 0.4) return 'unqualified';
      return 'warning';
    };

    const getEvaluationTagType = (evaluation) => {
      switch(evaluation) {
        case '表现优秀': return 'success';
        case '表现良好': return 'warning';
        case '学情预警': return 'danger';
        default: return '';
      }
    };

    const truncateName = (name) => {
      return name.length > 6 ? name.substring(0, 6) + '...' : name;
    };

    const updatePerformanceSummary = () => {
      let excellent = 0, good = 0, warning = 0;
      
      learnerInfo.value.domains.forEach(domain => {
        if (domain.evaluation === '表现优秀') excellent++;
        else if (domain.evaluation === '表现良好') good++;
        else if (domain.evaluation === '学情预警') warning++;
      });
      
      performanceItems.value = [
        { type: 'excellent', label: '表现优秀', count: excellent },
        { type: 'good', label: '表现良好', count: good },
        { type: 'warning', label: '学情预警', count: warning }
      ];
    };

    // 图表相关
    let overallChart = null;
    let domainChart = null;

    const initOverallChart = () => {
      if (!overallChart) {
        overallChart = echarts.init(document.querySelector('.performance-card .chart'));
      }
      
      const data = [
        { value: performanceItems.value[0].count, name: '表现优秀', itemStyle: { color: '#67C23A' } },
        { value: performanceItems.value[1].count, name: '表现良好', itemStyle: { color: '#E6A23C' } },
        { value: performanceItems.value[2].count, name: '学情预警', itemStyle: { color: '#F56C6C' } }
      ];
      
      overallChart.setOption({
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
      });
    };

    const initDomainChart = () => {
      if (!selectedDomain.value) return;
      
      if (!domainChart) {
        domainChart = echarts.init(document.querySelector('.domain-card .chart'));
      }
      
      const points = selectedDomain.value.knowledgePoints;
      let excellent = 0, good = 0, unqualified = 0, warning = 0;
      
      points.forEach(point => {
        if (point.score >= 0.8) excellent++;
        else if (point.score >= 0.6) good++;
        else if (point.score >= 0.4) unqualified++;
        else warning++;
      });
      
      const data = [];
      if (excellent > 0) data.push({ value: excellent, name: '优秀', itemStyle: { color: '#67C23A' } });
      if (good > 0) data.push({ value: good, name: '良好', itemStyle: { color: '#E6A23C' } });
      if (unqualified > 0) data.push({ value: unqualified, name: '不合格', itemStyle: { color: '#F56C6C' } });
      if (warning > 0) data.push({ value: warning, name: '严重不足', itemStyle: { color: '#909399' } });
      
      domainChart.setOption({
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
      });
    };

    const initCharts = () => {
      initOverallChart();
      initDomainChart();
    };

    const handleResize = () => {
      if (overallChart) overallChart.resize();
      if (domainChart) domainChart.resize();
    };

    // 生命周期
    onMounted(() => {
      initCharts();
      window.addEventListener('resize', handleResize);
      updatePerformanceSummary();
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

.dashboard-container {
  display: flex;
  gap: 20px;
}

.left-panel, .right-panel {
  flex: 1;
  min-width: 0; /* 防止flex元素溢出 */
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

.domains-list .el-tag {
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
.good { background-color: #E6A23C; }
.unqualified { background-color: #F56C6C; }
.warning { background-color: #909399; }

.knowledge-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}

.knowledge-item {
  display: flex;
  align-items: center;
  padding: 6px;
  background: #f5f7fa;
  border-radius: 4px;
  height: 32px;
  overflow: hidden;
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

.domain-evaluation {
  margin-top: 10px;
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

@media (max-width: 992px) {
  .dashboard-container {
    flex-direction: column;
  }
  
  .knowledge-grid {
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  }
}

@media (max-width: 768px) {
  .knowledge-grid {
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  }
}
</style>