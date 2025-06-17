export interface paginationConfigType {
  background: boolean
  small: boolean
  layout: string
  disabled: boolean
  page_sizes: number[]
}
// 分页参数
export interface paginationParamsType {
  page: number
  pageSize: number
  total: number
}
