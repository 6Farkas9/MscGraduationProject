<template>
  <WangEditor v-model="editVal" height="calc(100vh - 180px)" />
</template>

<script setup lang="ts">

const editVal = ref("Hello! WangEditor")
watchEffect(() => {
  console.log(editVal.value)
})
</script>

<style scoped></style>
