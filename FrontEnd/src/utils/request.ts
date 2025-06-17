import { message } from "@/Hooks/Element-plus"
import { useUserStore } from "@/stores/modules/user"

import axios from "axios"
import type { Method, AxiosRequestConfig } from "axios"
import { error_code } from "./error-code"
import { GETTOKEN } from "./local"
const service = axios.create({
  baseURL: import.meta.env.VITE_BASE_URL,
  timeout: 5000,
})
// 请求拦截器
service.interceptors.request.use((config) => {
  const userStore = useUserStore()
  if (GETTOKEN()) {
    config.headers.Token = GETTOKEN()
  }
  return config
})
// 必须先拦截一下
// 响应拦截器
service.interceptors.response.use(
  (response) => response.data,
  (err) => {
    message("error", error_code(err.response.status) || "网络问题")
    return Promise.reject(new Error(err))
  },
)
type Data<T> = {
  code: string
  message: string
  data: T
}
const request = <T>(
  url: string,
  method: Method = "get",
  data?: object | string,
  config?: AxiosRequestConfig,
) => {
  return service.request<T, Data<T>>({
    url,
    method,
    [method.toLowerCase() === "get" ? "params" : "data"]: data,
    ...config,
  })
}
export default request
