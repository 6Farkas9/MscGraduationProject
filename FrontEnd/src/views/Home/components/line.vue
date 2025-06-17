<template>
  <div class="line" ref="lineRef" />
</template>

<script setup lang="ts">
import * as echarts from "echarts"
const lineRef = ref()
onMounted(() => {})
nextTick(() => {
  let myChart = echarts.init(lineRef.value)

  let option = {
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        label: {
          backgroundColor: "#6a7985",
        },
      },
    },
    legend: {
      textStyle: {
        color: "gray",
      },
      top: 30,
      right: 10,
    },
    grid: {
      left: "3%",
      right: "6%",
      bottom: "3%",
      containLabel: true,
    },
    xAxis: [
      {
        type: "category",
        boundaryGap: false,
        data: ["Vue", "Pina", "Element-plus", "Vite", "Echarts", "TypeScript"],
        axisLabel: {
          // color: "black",
          interval: 0,
        },
        axisTick: {
          show: false,
        },
      },
    ],
    yAxis: [
      {
        type: "value",
        axisLabel: {
          // color: "white",
        },
      },
    ],
    series: [
      {
        name: "日常访问量",
        type: "line",
        smooth: true,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(
            0,
            0,
            0,
            1,
            [
              {
                offset: 0,
                color: "rgba(8, 171, 231, 0.8)", // 渐变色的起始颜色
              },
              {
                offset: 1,
                color: "rgba(8, 171, 231, 0.2)", // 渐变线的结束颜色
              },
            ],
            false,
          ),
          shadowColor: "rgba(0, 0, 0, 0.8)",
        },
        symbol: "circle",
        symbolSize: 5,
        showSymbol: false,
        itemStyle: {
          color: "#08abe7",
        },
        lineStyle: {
          color: "#08abe7",
        },

        emphasis: {
          focus: "series",
        },
        data: [260, 200, 400, 350, 600, 430],
      },
      {
        name: "节假日访问量",
        type: "line",
        smooth: true,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(
            0,
            0,
            0,
            1,
            [
              {
                offset: 0,
                color: "rgba(8, 218, 155, 0.8)", // 渐变色的起始颜色
              },
              {
                offset: 1,
                color: "rgba(8, 218, 155, 0.2)", // 渐变线的结束颜色
              },
            ],
            false,
          ),
          shadowColor: "rgba(0, 0, 0, 0.8)",
        },
        // 点特效
        symbol: "circle",
        symbolSize: 5,
        showSymbol: false,
        itemStyle: {
          color: "#08da9bcc",
          borderColor: "#08da9bcc",
        },
        lineStyle: {
          color: "#08da9bcc",
        },
        emphasis: {
          focus: "series",
        },
        data: [620, 400, 882, 691, 734, 890],
      },
    ],
  }
  window.addEventListener("resize", () => {
    myChart.resize()
  })
  option && myChart.setOption(option)
})
</script>

<style scoped>
.box {
  color: rgba(8, 171, 231, 0.8);
}
</style>
