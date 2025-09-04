#!/usr/bin/env python3
"""
Enhanced XYZ to SVG converter with improved chemical correctness
Based on advanced RDKit strategies for robust molecular structure generation
"""

import argparse
import csv
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, inchi, rdCoordGen, rdDetermineBonds, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize

# 静音 RDKit 警告
RDLogger.DisableLog('rdApp.warning')
rdDepictor.SetPreferCoordGen(True)

def parse_charge_from_xyz_comment(xyz_path):
    """解析xyz文件注释行中的电荷信息"""
    try:
        with open(xyz_path, 'r', encoding='utf-8', errors='ignore') as f:
            _ = f.readline()
            comment = f.readline().strip()
        m = re.search(r'charge\s*[:=]\s*([+-]?\d+)', comment, flags=re.I)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0

def sanitize_soft(mol):
    """分步消毒：哪步炸在了哪步我们都知道；并尽量把能修的修掉"""
    try:
        Chem.SanitizeMol(mol)  # 一步过当然最好
        return mol, None
    except Exception as e_all:
        # 分步来
        flags_seq = [
            Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
            Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            Chem.SanitizeFlags.SANITIZE_CLEANUP,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
            Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION,
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
        ]
        for flg in flags_seq:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=flg)
            except Exception:
                # 不中断，继续下一步，让最大子集通过
                pass
        # 最后再尝试一次全量
        try:
            Chem.SanitizeMol(mol)
            return mol, None
        except Exception as e_final:
            return mol, f"soft-sanitize residual issue: {e_final}"

def _fix_hypervalent_hydrogens(mol):
    """修复氢原子多价问题：只保留离它最近的重原子那根键"""
    if mol.GetNumConformers() == 0:
        return mol
    conf = mol.GetConformer()
    rw = Chem.RWMol(mol)
    to_remove = []
    
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1:
            continue
        nbrs = [n.GetIdx() for n in a.GetNeighbors()]
        if len(nbrs) <= 1:
            continue
        
        # 只保留与最近"重原子"的那根键
        dists = []
        pa = conf.GetAtomPosition(a.GetIdx())
        for n in nbrs:
            pn = conf.GetAtomPosition(n)
            d = (pa - pn).Length()
            dists.append((d, n, mol.GetAtomWithIdx(n).GetAtomicNum()))
        
        # 优先重原子，再按距离最短
        dists.sort(key=lambda x: (x[2]==1, x[0]))
        keep_n = dists[0][1]
        for n in nbrs:
            if n != keep_n:
                to_remove.append((a.GetIdx(), n))
    
    # 移除多余的键
    for i, j in to_remove:
        rw.RemoveBond(i, j)
    return rw.GetMol()

def _manual_bond_assignment(raw_mol):
    """更安全的距离阈值连键：禁止H-H；先重-重，再重-H；按元素对阈值"""
    rw = Chem.RWMol(raw_mol)
    conf = rw.GetConformer()

    # 移除现有键
    for b in list(rw.GetBonds()):
        rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    def is_heavy(a): 
        return a.GetAtomicNum() > 1

    # 基础共价半径（Å）
    rcov = {
        1: 0.31,  6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
        15: 1.07, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39
    }

    def rcov_sum(i, j):
        ai, aj = rw.GetAtomWithIdx(i), rw.GetAtomWithIdx(j)
        ri = rcov.get(ai.GetAtomicNum(), 0.9)
        rj = rcov.get(aj.GetAtomicNum(), 0.9)
        return ri + rj

    def dist(i, j):
        pi = conf.GetAtomPosition(i)
        pj = conf.GetAtomPosition(j)
        return (pi - pj).Length()

    N = rw.GetNumAtoms()
    pairs = []
    
    for i in range(N):
        for j in range(i+1, N):
            ai = rw.GetAtomWithIdx(i)
            aj = rw.GetAtomWithIdx(j)
            dij = dist(i, j)

            # 禁止 H-H（除非整分子只有两个H）
            if ai.GetAtomicNum()==1 and aj.GetAtomicNum()==1:
                if N==2:  # H2 特例
                    pairs.append((dij, i, j, "HH"))
                continue

            s = rcov_sum(i, j)
            # 按元素配对给不同的松紧：重-重最松、重-H 次之
            if is_heavy(ai) and is_heavy(aj):
                thr = 1.25 * s
            else:
                thr = 1.15 * s  # X-H 稍紧
            
            if dij < thr:
                tag = "HH" if (ai.GetAtomicNum()==1 and aj.GetAtomicNum()==1) else "OK"
                pairs.append((dij, i, j, tag))

    # 先连重-重，再连重-H，按距离从短到长
    pairs.sort(key=lambda x: (x[3]!="OK", x[0]))
    
    for _, i, j, _ in pairs:
        ai = rw.GetAtomWithIdx(i)
        aj = rw.GetAtomWithIdx(j)
        
        # 若已经相连则跳过
        if rw.GetBondBetweenAtoms(i, j): 
            continue
            
        # 简单的"最大价键"守卫
        max_degree = {8: 2, 7: 4, 6: 4, 16: 6, 15: 5}
        for aidx in (i, j):
            a = rw.GetAtomWithIdx(aidx)
            md = max_degree.get(a.GetAtomicNum(), 6)
            if len(a.GetNeighbors()) >= md:
                break
        else:
            rw.AddBond(i, j, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    mol, _ = sanitize_soft(mol)
    return mol

def _try_determine_bonds_with_fallback(raw_mol, charge_hint):
    """增强的回退策略：多种参数组合 + 温和消毒"""
    attempts = [
        # 标准策略
        {"name": "Hueckel+charge", "useHueckel": True, "charge": charge_hint, "allowChargedFragments": True},
        {"name": "Hueckel-nocharge", "useHueckel": True, "allowChargedFragments": True},
        {"name": "NoHueckel+charge", "useHueckel": False, "charge": charge_hint, "allowChargedFragments": True},
        {"name": "NoHueckel-nocharge", "useHueckel": False, "allowChargedFragments": True},
        
        # 强制中性策略
        {"name": "Hueckel+neutral", "useHueckel": True, "charge": 0, "allowChargedFragments": False},
        {"name": "Hueckel-neutral", "useHueckel": True, "allowChargedFragments": False},
        {"name": "NoHueckel+neutral", "useHueckel": False, "charge": 0, "allowChargedFragments": False},
        {"name": "NoHueckel-neutral", "useHueckel": False, "allowChargedFragments": False},
    ]
    
    last_err = None
    for attempt in attempts:
        try:
            tmp = Chem.Mol(raw_mol)
            kw = {k: v for k, v in attempt.items() if k != "name"}
            rdDetermineBonds.DetermineBonds(tmp, embedChiral=True, **kw)
            
            # 修复氢的多价，再温和消毒
            tmp = _fix_hypervalent_hydrogens(tmp)
            tmp, _ = sanitize_soft(tmp)
            return tmp, None, attempt["name"]
        except Exception as e:
            last_err = e
            if "charge" in str(e).lower() and "does not match" in str(e).lower():
                continue

    # 最后的救命稻草：手动连键
    try:
        m = _manual_bond_assignment(raw_mol)
        return m, None, "manual_bonds"
    except Exception:
        return None, f"All strategies failed. Last error: {str(last_err)}", "failed"

def mol_from_xyz_with_bonds(xyz_path, ignore_charge_hint=False):
    """增强的 XYZ → 分子转换，支持多种回退策略"""
    try:
        raw = Chem.MolFromXYZFile(str(xyz_path))
        if raw is None:
            return None, "Failed to read XYZ file", "read_failed"

        mol0 = Chem.Mol(raw)
        charge = 0 if ignore_charge_hint else parse_charge_from_xyz_comment(xyz_path)

        # TS文件特殊处理
        name_lower = str(xyz_path).lower()
        prefer_loose = ("_ts" in name_lower) or ("-ts" in name_lower)

        if prefer_loose:
            m, err, strategy = _try_determine_bonds_with_fallback(mol0, charge_hint=None)
        else:
            m, err, strategy = _try_determine_bonds_with_fallback(mol0, charge_hint=charge)

        if m is None:
            return None, err, strategy

        # 标准化
        m = rdMolStandardize.Cleanup(m)
        m = rdMolStandardize.Reionize(m)
        m = rdMolStandardize.DisconnectOrganometallics(m)

        # 从3D赋予立体化学
        Chem.AssignStereochemistryFrom3D(m)
        return m, None, strategy

    except Exception as e:
        return None, str(e), "exception"

def draw_svg(mol, w=512, h=384, legend="", ring_templates=None):
    """改进的SVG绘制，支持环模板和更好的ACS风格"""
    try:
        if ring_templates and os.path.exists(ring_templates):
            try:
                rdDepictor.AddRingSystemTemplates(ring_templates)
            except Exception:
                pass

        rdCoordGen.AddCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        try:
            opts.bondLineWidth = 2
        except:
            pass
        try:
            opts.minFontSize = 0.8
        except:
            pass

        # ACS 1996风格 - 修复API调用
        try:
            rdMolDraw2D.SetACS1996Mode(opts)
        except:
            pass

        # 如果kekulize失败，用圆圈芳香
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except:
            for b in mol.GetBonds():
                if b.GetIsAromatic():
                    opts.useCircularHighlight = True
                    break

        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
        drawer.FinishDrawing()
        return drawer.GetDrawingText(), None
    except Exception as e:
        return None, str(e)

def find_xyz_files(base_dir=".", pattern=r'MR_\d+_.*\.xyz$'):
    """查找匹配模式的xyz文件"""
    xyz_files = []
    regex = re.compile(pattern)

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if regex.match(file):
                xyz_files.append(os.path.join(root, file))

    return sorted(xyz_files)

def process_single_xyz(xyz_path, out_dir, args, qc_writer=None):
    """处理单个xyz文件"""
    xyz_file = Path(xyz_path)
    rel_path = os.path.relpath(xyz_path, ".")

    # 创建分子
    mol, error, strategy = mol_from_xyz_with_bonds(
        xyz_path,
        ignore_charge_hint=args.ignore_charge_hint
    )

    if mol is None:
        if qc_writer:
            qc_writer.writerow([xyz_file.name, "FAIL", "", "", strategy, str(error)])
        return False, f"Failed to create molecule ({strategy}): {error}"

    try:
        # 生成SMILES和InChIKey
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        try:
            inchi_key = inchi.MolToInchiKey(mol)
        except:
            inchi_key = ""

        # 保存SDF
        sdf_file = out_dir / (xyz_file.stem + ".sdf")
        Chem.MolToMolFile(mol, str(sdf_file))

        # 保存SMILES
        smiles_file = out_dir / (xyz_file.stem + ".smiles")
        smiles_file.write_text(smiles, encoding="utf-8")

        # 生成SVG
        svg_size = args.svg_size.split('x') if 'x' in args.svg_size else [512, 384]
        w, h = int(svg_size[0]), int(svg_size[1])

        svg, svg_error = draw_svg(
            mol,
            w=w, h=h,
            legend=xyz_file.stem,
            ring_templates=args.ring_templates
        )

        if svg is None:
            if qc_writer:
                qc_writer.writerow([xyz_file.name, "SVG_FAIL", smiles, inchi_key, strategy, str(svg_error)])
            return False, f"SVG generation failed ({strategy}): {svg_error}"

        svg_file = out_dir / (xyz_file.stem + ".svg")
        svg_file.write_text(svg, encoding="utf-8")

        if qc_writer:
            qc_writer.writerow([xyz_file.name, "SUCCESS", smiles, inchi_key, strategy, ""])

        return True, f"Success ({strategy}): {smiles}"

    except Exception as e:
        if qc_writer:
            qc_writer.writerow([xyz_file.name, "ERROR", "", "", strategy, str(e)])
        return False, f"Processing error ({strategy}): {str(e)}"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Enhanced XYZ to SVG converter with robust chemical processing"
    )
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count(),
                       help="Number of parallel jobs (default: CPU count)")
    parser.add_argument("--ignore-charge-hint", action="store_true",
                       help="Ignore charge hints from XYZ comments")
    parser.add_argument("--ring-templates", type=str,
                       help="Path to ring templates JSON file")
    parser.add_argument("--svg-size", type=str, default="512x384",
                       help="SVG size as WIDTHxHEIGHT (default: 512x384)")
    parser.add_argument("--output-dir", "-o", type=str, default="enhanced_structures",
                       help="Output directory (default: enhanced_structures)")
    parser.add_argument("--pattern", type=str, default=r'MR_\d+_.*\.xyz$',
                       help="Regex pattern for XYZ files (default: MR_*_*.xyz)")
    parser.add_argument("--base-dir", type=str, default=".",
                       help="Base directory to search for XYZ files (default: current)")

    args = parser.parse_args()

    print("🧪 Enhanced XYZ to SVG Converter")
    print("=" * 60)
    print(f"⚙️  Configuration:")
    print(f"   Parallel jobs: {args.jobs}")
    print(f"   SVG size: {args.svg_size}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Ignore charge hints: {args.ignore_charge_hint}")
    if args.ring_templates:
        print(f"   Ring templates: {args.ring_templates}")
    print()

    # 查找文件
    xyz_files = find_xyz_files(args.base_dir, args.pattern)

    if not xyz_files:
        print(f"❌ No XYZ files found matching pattern: {args.pattern}")
        return

    print(f"📁 Found {len(xyz_files)} XYZ files")

    # 显示示例
    print(f"\n📋 Example files:")
    for i, xyz_file in enumerate(xyz_files[:5]):
        rel_path = os.path.relpath(xyz_file, args.base_dir)
        print(f"   {i+1}. {rel_path}")
    if len(xyz_files) > 5:
        print(f"   ... and {len(xyz_files) - 5} more files")

    # 确认
    response = input(f"\n🤔 Process {len(xyz_files)} files with {args.jobs} parallel jobs? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return

    # 创建输出目录
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 创建QC报表
    qc_path = out_dir / "qc_report.csv"

    print(f"\n📁 Output directory: {out_dir}")
    print(f"📊 QC report: {qc_path}")
    print(f"🚀 Starting processing with {args.jobs} workers...")
    print("=" * 60)

    # 统计
    stats = {"success": 0, "failed": 0}
    strategy_stats = {}

    # 并行处理
    with open(qc_path, 'w', newline='', encoding='utf-8') as qc_file:
        qc_writer = csv.writer(qc_file)
        qc_writer.writerow(["file", "status", "smiles", "inchikey", "strategy", "note"])

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            # 提交所有任务
            futures = {
                executor.submit(process_single_xyz, xyz_file, out_dir, args, qc_writer): xyz_file
                for xyz_file in xyz_files
            }

            # 处理结果
            for i, future in enumerate(as_completed(futures), 1):
                xyz_file = futures[future]
                rel_path = os.path.relpath(xyz_file, args.base_dir)

                try:
                    success, message = future.result()
                    if success:
                        stats["success"] += 1
                        # 提取策略信息
                        if "(" in message and ")" in message:
                            strategy = message.split("(")[1].split(")")[0]
                            strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
                        print(f"✅ [{i}/{len(xyz_files)}] {rel_path}: {message}")
                    else:
                        stats["failed"] += 1
                        print(f"❌ [{i}/{len(xyz_files)}] {rel_path}: {message}")
                except Exception as e:
                    stats["failed"] += 1
                    print(f"🔥 [{i}/{len(xyz_files)}] {rel_path}: Exception - {str(e)}")

                # 进度更新
                if i % 100 == 0:
                    progress = (i / len(xyz_files)) * 100
                    print(f"\n📊 Progress: {i}/{len(xyz_files)} ({progress:.1f}%) - Success: {stats['success']}, Failed: {stats['failed']}")

    # 最终总结
    total_files = len(xyz_files)
    success_rate = (stats['success'] / total_files) * 100 if total_files > 0 else 0

    print("\n" + "=" * 80)
    print("🎉 ENHANCED PROCESSING COMPLETED!")
    print("=" * 80)
    print(f"📊 Final Results:")
    print(f"   Total files:       {total_files}")
    print(f"   Successful:        {stats['success']} ✅")
    print(f"   Failed:            {stats['failed']} ❌")
    print(f"   Success rate:      {success_rate:.1f}%")

    if strategy_stats:
        print(f"\n🔧 Strategy Usage:")
        for strategy, count in sorted(strategy_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['success']) * 100 if stats['success'] > 0 else 0
            print(f"   {strategy}: {count} ({percentage:.1f}%)")

    if stats['success'] > 0:
        print(f"\n📁 Output files saved to: {out_dir}")
        sdf_count = len(list(out_dir.glob("*.sdf")))
        smiles_count = len(list(out_dir.glob("*.smiles")))
        svg_count = len(list(out_dir.glob("*.svg")))

        print(f"   Generated files:")
        print(f"     SDF files:       {sdf_count}")
        print(f"     SMILES files:    {smiles_count}")
        print(f"     SVG files:       {svg_count}")

    print(f"\n📋 Detailed QC report: {qc_path}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except ImportError as e:
        print("❌ Missing dependencies!")
        print("Please install RDKit: conda install -c conda-forge rdkit")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
